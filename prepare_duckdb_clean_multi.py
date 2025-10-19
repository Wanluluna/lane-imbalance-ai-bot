
# prepare_duckdb_clean_multi.py
import os
import json
import duckdb

# =========================
# Config (edit paths if needed)
# =========================
LANE_SRC = r"C:\Users\wli0001\Downloads\Lane Imbalance Data_R13.csv"
TONNAGE_SRC = r"C:\Users\wli0001\Downloads\Target Tonnage Data_R13"  # extension optional (.csv/.parquet)
DB_PATH = r"C:\Users\wli0001\Lane Imbalance\lane.duckdb"

# Requested table names
LANE_RAW = "lane_imbalance_raw"
TONNAGE_RAW = "target_tonnage_raw"
LANE_CLEAN = "lane_imbalance_clean"
TONNAGE_CLEAN = "target_tonnage_clean"
LANE_NUM = "lane_imbalance_num"
TONNAGE_NUM = "target_tonnage_num"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# =========================
# Utils
# =========================
def _sql(s: str) -> str:
    return s.replace("'", "''")

def _ensure_ext(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext == "":
        return base + ".csv"
    return path

def csv_to_parquet(src_path: str) -> str:
    """
    Convert CSV -> Parquet via DuckDB COPY for speed & stability.
    If input already Parquet, return as-is.
    """
    src_path = _ensure_ext(src_path)
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in [".csv", ".parquet"]:
        raise ValueError(f"Only CSV/Parquet supported. Got: {src_path}")
    if ext == ".parquet":
        return src_path

    pq = os.path.splitext(src_path)[0] + ".parquet"
    need = (not os.path.exists(pq)) or (os.path.getmtime(src_path) > os.path.getmtime(pq))
    if need:
        print(f"[CSV→Parquet] {src_path} -> {pq}")
        con = duckdb.connect()
        con.execute(f"""
        COPY (
          SELECT * FROM read_csv_auto('{_sql(src_path)}', sample_size=-1)
        ) TO '{_sql(pq)}' (FORMAT PARQUET);
        """)
        con.close()
    else:
        print(f"[Parquet cached] {pq}")
    return pq

def sanitize_alias(name: str) -> str:
    import re
    return re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_').lower()


# =========================
# Builders
# =========================
def build_clean_table(con, src_table: str, out_table: str, forced_dtypes: dict):
    """
    Create a CLEAN table with proper dtypes based on forced_dtypes rules.
    Supported dtype keywords: VARCHAR, INTEGER, DOUBLE, PERCENT
      - VARCHAR : trim text, '' -> NULL
      - INTEGER : strip non [0-9-], TRY_CAST to INTEGER
      - DOUBLE  : strip non [0-9.-], TRY_CAST to DOUBLE
      - PERCENT : remove symbols incl. '%', TRY_CAST to DOUBLE, then /100.0 (e.g. '67%' -> 0.67)
    Unspecified columns default to VARCHAR cleaning.
    """

    # Safe macros: always CAST to VARCHAR first to avoid TRIM(BOOLEAN) etc.
# --- hardened cleaning macros (replace the four macros in build_clean_table) ---
    con.execute("""
    CREATE OR REPLACE MACRO CLEAN_TEXT(x) AS
      CASE
        WHEN x IS NULL THEN NULL
        ELSE NULLIF(TRIM(CAST(x AS VARCHAR)), '')
      END;
    """)
    
    con.execute("""
    CREATE OR REPLACE MACRO CLEAN_INTEGER(x) AS
      CASE
        WHEN x IS NULL THEN NULL
        ELSE
          CASE
            WHEN NULLIF(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9\\-]', '', 'g'), '') IS NULL
              THEN NULL
            ELSE CAST(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9\\-]', '', 'g') AS INTEGER)
          END
      END;
    """)
    
    con.execute("""
    CREATE OR REPLACE MACRO CLEAN_DOUBLE(x) AS
      CASE
        WHEN x IS NULL THEN NULL
        ELSE
          CASE
            WHEN NULLIF(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9.\\-]', '', 'g'), '') IS NULL
              THEN NULL
            ELSE CAST(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9.\\-]', '', 'g') AS DOUBLE)
          END
      END;
    """)
    
    con.execute("""
    CREATE OR REPLACE MACRO CLEAN_PERCENT(x) AS
      CASE
        WHEN x IS NULL THEN NULL
        ELSE
          CASE
            WHEN NULLIF(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9.\\-]', '', 'g'), '') IS NULL
              THEN NULL
            ELSE CAST(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9.\\-]', '', 'g') AS DOUBLE) / 100.0
          END
      END;
    """)
    

    schema_df = con.execute(f"DESCRIBE SELECT * FROM {src_table} LIMIT 0").df()
    cols = list(schema_df["column_name"])
    select_exprs = []

    for c in cols:
        dtype = forced_dtypes.get(c)
        if dtype == "INTEGER":
            expr = f"CLEAN_INTEGER(\"{c}\") AS \"{c}\""
        elif dtype == "DOUBLE":
            expr = f"CLEAN_DOUBLE(\"{c}\") AS \"{c}\""
        elif dtype == "PERCENT":
            expr = f"CLEAN_PERCENT(\"{c}\") AS \"{c}\""
        elif dtype == "VARCHAR":
            expr = f"CLEAN_TEXT(\"{c}\") AS \"{c}\""
        else:
            # default: treat as text
            expr = f"CLEAN_TEXT(\"{c}\") AS \"{c}\""
        select_exprs.append(expr)

    con.execute(f"DROP TABLE IF EXISTS {out_table};")
    con.execute(f"""
    CREATE TABLE {out_table} AS
    SELECT {", ".join(select_exprs)} FROM {src_table};
    """)


def build_num_table(con, base_table: str, out_table: str):
    """
    Build a numeric-augmented table for convenience/compatibility:
    Adds *_num DOUBLE columns for names that look numeric.
    """
    con.execute("""
    CREATE OR REPLACE MACRO SAFE_TO_DOUBLE(x) AS
      COALESCE(
        TRY_CAST(REGEXP_REPLACE(TRIM(CAST(x AS VARCHAR)), '[^0-9.\\-]', '', 'g') AS DOUBLE),
        0.0
      );
    """)
    schema_df = con.execute(f"DESCRIBE SELECT * FROM {base_table} LIMIT 0").df()
    cols = list(schema_df["column_name"])
    add_exprs = []
    patterns = [
        "volume", "weight", "case", "cost", "amount", "price",
        "qty", "distance", "score", "ton", "target", "revenue", "profit"
    ]
    for c in cols:
        lc = c.lower()
        if any(p in lc for p in patterns):
            add_exprs.append(f'SAFE_TO_DOUBLE("{c}") AS {sanitize_alias(c)}_num')

    sel = ["*"] + add_exprs
    con.execute(f"DROP TABLE IF EXISTS {out_table};")
    con.execute(f"CREATE TABLE {out_table} AS SELECT {', '.join(sel)} FROM {base_table};")


# =========================
# Main
# =========================
def main():
    lane_pq = csv_to_parquet(LANE_SRC)
    tonnage_pq = csv_to_parquet(TONNAGE_SRC)

    print(f"[Build DB] {DB_PATH}")
    con = duckdb.connect(DB_PATH)
    con.execute("PRAGMA threads=4;")

    # 1) RAW tables (requested names)
    print("→ Creating RAW tables")
    con.execute(f"DROP TABLE IF EXISTS {LANE_RAW};")
    con.execute(f"CREATE TABLE {LANE_RAW} AS SELECT * FROM read_parquet('{_sql(lane_pq)}');")
    con.execute(f"DROP TABLE IF EXISTS {TONNAGE_RAW};")
    con.execute(f"CREATE TABLE {TONNAGE_RAW} AS SELECT * FROM read_parquet('{_sql(tonnage_pq)}');")

    # 2) Forced dtype mappings from your specs
    # Lane Imbalance
    forced_dtypes_lane = {
        "Year": "VARCHAR",
        "Week": "VARCHAR",
        "Period": "VARCHAR",
        "Load Count": "INTEGER",
        "Weight (KG)": "DOUBLE",
        "Volume (Cubic Feet)": "DOUBLE",
        "Linehaul Cost": "DOUBLE",
        "Distance (Miles)": "DOUBLE",
        "Zone Score - Annual": "DOUBLE",
        "Zone Score - Period": "DOUBLE",
        "Province Score - Annual": "DOUBLE",
        "Province Score - Period": "DOUBLE",
    }

    # Target Tonnage
    # 1) text-like even if numeric-looking (filters)
    forced_text_tonnage = [
        "Vendor", "BU#", "Customer ID", "R13 Frequency",
        "Shipping Location", "DC", "Freight Class"
    ]
    # 2) integers
    forced_int_tonnage = ["PO Count", "Weight (KG)", "Weight(LBS)", "R13 Province Score"]
    # 3) decimals (money)
    forced_double_tonnage = ["Estimated Revenue", "Estimated Profit"]
    # 4) percentages '67%' -> 0.67
    forced_percent_tonnage = ["DED%", "DDP%"]

    forced_dtypes_tonnage = {}
    for c in forced_text_tonnage:
        forced_dtypes_tonnage[c] = "VARCHAR"
    for c in forced_int_tonnage:
        forced_dtypes_tonnage[c] = "INTEGER"
    for c in forced_double_tonnage:
        forced_dtypes_tonnage[c] = "DOUBLE"
    for c in forced_percent_tonnage:
        forced_dtypes_tonnage[c] = "PERCENT"

    # 3) CLEAN (typed) tables
    print("→ Building CLEAN (typed) tables")
    build_clean_table(con, LANE_RAW, LANE_CLEAN, forced_dtypes_lane)
    build_clean_table(con, TONNAGE_RAW, TONNAGE_CLEAN, forced_dtypes_tonnage)

    # 4) NUM (compatibility) tables
    print("→ Building NUMERIC (compat) tables")
    build_num_table(con, LANE_CLEAN, LANE_NUM)
    build_num_table(con, TONNAGE_CLEAN, TONNAGE_NUM)

    # 5) ANALYZE + shape cache (for *_clean)
    print("→ ANALYZE & shape cache")
    for t in [LANE_RAW, TONNAGE_RAW, LANE_CLEAN, TONNAGE_CLEAN, LANE_NUM, TONNAGE_NUM]:
        con.execute(f"ANALYZE {t};")

    shapes = {}
    for t in [LANE_CLEAN, TONNAGE_CLEAN]:
        rows = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        cols = len(con.execute(f"DESCRIBE SELECT * FROM {t} LIMIT 0").df())
        shapes[t] = {"rows": int(rows), "cols": int(cols)}

    cache_dir = os.path.dirname(DB_PATH)
    with open(os.path.join(cache_dir, "shape_cache.json"), "w", encoding="utf-8") as f:
        json.dump(shapes, f, indent=2)

    con.close()
    print("✅ All done!")
    print(f"DuckDB: {DB_PATH}")
    print(f"Shape cache: {os.path.join(cache_dir, 'shape_cache.json')}")


if __name__ == "__main__":
    main()
