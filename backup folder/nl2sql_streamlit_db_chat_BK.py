# nl2sql_streamlit_db_chat.py
import os
import json
import re
import duckdb
import streamlit as st
import math

# ===================== Optional: Gemini =====================
HAS_GENAI = True
try:
    import google.generativeai as genai
except Exception:
    HAS_GENAI = False

MODEL = "models/gemini-2.0-flash-lite"  # fast & free-tier friendly
API_KEY = "AIzaSyA-QzLpmqi6HOYiuhDrfglwZg4Tf-6I-js".strip()
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if HAS_GENAI and API_KEY:
    genai.configure(api_key=API_KEY)

# ===================== Config =====================
DB_PATH = r"C:\Users\wli0001\Lane Imbalance\lane.duckdb"
TABLE_OPTIONS = [
    ("Lane Imbalance", "lane_imbalance_clean"),
    ("Target Tonnage", "target_tonnage_clean"),
]
DEFAULT_TABLE = "lane_imbalance_clean"

# ===================== Safe set options =====================
def _set_option_safe(key: str, value):
    try:
        st.set_option(key, value)
    except Exception:
        pass

# Reduce accidental reloads; ignore unsupported keys
_set_option_safe("server.fileWatcherType", "none")

# ===================== Page =====================
st.set_page_config(page_title="Lane Imbalance Data_AI Bot", layout="wide")
st.title("⚙️📈 Lane Imbalance Data_AI Bot")
st.caption("💬 Ask questions about the dataset")

# ===================== Connect (cached) =====================
@st.cache_resource(show_spinner=False)
def get_con(db_path: str):
    # read-only prevents accidental writes + slightly safer
    return duckdb.connect(db_path, read_only=True)

con = get_con(DB_PATH)

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Settings")

    dataset_label = st.selectbox(
        "Dataset",
        options=TABLE_OPTIONS,
        index=0,
        format_func=lambda x: x[0],
        key="dataset_selector",
    )
    selected_label, selected_table = dataset_label

    # shape from cache; fallback to DB (fast)
    rows, cols = None, None
    cache_path = os.path.join(os.path.dirname(DB_PATH), "shape_cache.json")
    if os.path.exists(cache_path):
        try:
            shapes = json.load(open(cache_path, "r", encoding="utf-8"))
            sc = shapes.get(selected_table)
            if sc:
                rows, cols = sc.get("rows"), sc.get("cols")
        except Exception:
            pass
    if rows is None or cols is None:
        try:
            rows = int(con.execute(f"SELECT COUNT(*) FROM {selected_table};").fetchone()[0])
            cols = len(con.execute(f"DESCRIBE SELECT * FROM {selected_table} LIMIT 0").df())
        except Exception:
            rows, cols = 0, 0

    st.markdown(f"📊 **Shape:** {rows:,} rows × {cols} columns")

    st.divider()
    st.markdown("**Gemini model**")
    st.code(MODEL)

    st.markdown("**API key source**")
    st.write("Hard-coded" if API_KEY else "Environment variable or (missing)")
    if not HAS_GENAI:
        st.warning("google-generativeai not installed; only manual SQL will work.", icon="⚠️")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.experimental_rerun()

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def _schema_for_prompt(table: str):
    # Small peeks only, to keep the page snappy
    schema_df = con.execute(f"DESCRIBE SELECT * FROM {table} LIMIT 0").df()
    schema = {row["column_name"]: row["column_type"] for _, row in schema_df.iterrows()}
    head_df = con.execute(f"SELECT * FROM {table} LIMIT 6").df()
    return schema, head_df

def clean_sql(sql: str) -> str:
    s = (sql or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    s = re.sub(r"^`+|`+$", "", s).strip()
    return s

def try_execute_sql(sql: str, limit_rows: int = 1000):
    df = con.execute(sql).df()
    head = df.head(limit_rows)
    return {
        "rows": head.to_dict(orient="records"),
        "columns": list(head.columns),
        "total_rows": len(df),
        "df": df,
    }

def is_topn_query(text: str):
    """
    Detect 'top N' intent. Returns (True, N) or (False, None).
    Matches: 'top 3', 'Top10', '前 5', '前10名' (basic Chinese support).
    """
    if not text:
        return (False, None)
    m = re.search(r'(?:top\s*-?\s*(\d+))|(?:前\s*(\d+))', text, flags=re.IGNORECASE)
    if m:
        n = m.group(1) or m.group(2)
        try:
            return (True, int(n))
        except Exception:
            return (True, None)
    return (False, None)

def calc_table_height(num_rows: int, header_px: int = 38, row_px: int = 36,
                      min_px: int = 110, max_px: int = 420) -> int:
    """
    Compute a compact height so no big blank area appears.
    """
    h = header_px + max(1, num_rows) * row_px + 8  # +8 small padding
    return max(min(h, max_px), min_px)




def plan_sql(question: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        # Fallback if Gemini not available
        return f"SELECT * FROM {table} LIMIT 50;"
    schema, head_df = _schema_for_prompt(table)
    system_prompt = f"""
You are a senior data analyst. Generate **valid DuckDB SQL** to answer business questions.
Use the table `{table}` only. Only SELECT queries.
Return only the SQL (optionally inside ```sql code fences).

Data schema:
{json.dumps(schema, indent=2)}

Sample rows (CSV):
{head_df.to_csv(index=False)}

Guidelines:
- Use ORDER BY + LIMIT for top/bottom.
- Use GROUP BY with aggregations when needed (SUM, AVG, COUNT).
- Use exact column names; avoid ambiguity.
"""
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content([system_prompt, f"Question: {question}\nWrite the SQL now."])
    return (resp.text or "").strip()

def repair_sql(question: str, prev_sql: str, error_msg: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"SELECT * FROM {table} LIMIT 50;"
    schema, head_df = _schema_for_prompt(table)
    repair_prompt = f"""
The previous SQL failed. FIX it.

Question:
{question}

Failed SQL:
{prev_sql}

DuckDB Error:
{error_msg}

Data schema:
{json.dumps(schema, indent=2)}

Sample rows (CSV):
{head_df.to_csv(index=False)}

Rules:
- Return corrected SQL only.
- Use columns that actually exist.
"""
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content(repair_prompt)
    return (resp.text or "").strip()

def summarize(question: str, used_sql: str, result: dict) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"- Returned {result['total_rows']} rows\n- SQL: {used_sql}"
    model = genai.GenerativeModel(MODEL)
    resp = genai.GenerativeModel(MODEL).generate_content(f"""
Question: {question}

SQL used:
{used_sql}

First {min(10, len(result['df']))} rows shown; total_rows={result['total_rows']}.
Write concise bullet points that answer the question.
""")
    return (resp.text or "").strip()

def answer(question: str, table: str, limit_rows: int = 1000):
    sql1 = clean_sql(plan_sql(question, table))
    try:
        result = try_execute_sql(sql1, limit_rows=limit_rows)
        used_sql = sql1
    except Exception as e:
        sql2 = clean_sql(repair_sql(question, sql1, str(e), table))
        result = try_execute_sql(sql2, limit_rows=limit_rows)
        used_sql = sql2
    analysis = summarize(question, used_sql, result)
    return {"sql": used_sql, **result, "analysis": analysis}

# ===================== Chat state =====================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content} and {role, payload, table}


# Render chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**YOU** · Dataset: `{msg.get('display', msg.get('table',''))}`")
            st.markdown(msg["content"])
    else:
        # find the nearest previous user message to read their question
        prev_q = ""
        for j in range(i - 1, -1, -1):
            if st.session_state.messages[j]["role"] == "user":
                prev_q = st.session_state.messages[j]["content"]
                break

        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**")
            st.code(msg["payload"]["sql"], language="sql")

            # Prepare a compact dataframe for display
            render_df = msg["payload"]["df"].copy()

            # Add Rank if user asked for top-N
            topn, n = is_topn_query(prev_q)
            if topn:
                # if model accidentally returns >N, we still rank all returned rows
                render_df.insert(0, "Rank", range(1, len(render_df) + 1))

            height = calc_table_height(len(render_df))
            st.write(f"Total rows: {msg['payload']['total_rows']}")
            st.dataframe(render_df, use_container_width=True, height=height, hide_index=True)

            st.markdown(msg["payload"]["analysis"] or "_(no analysis)_")
            csv_bytes = render_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
                key=f"dl_{id(msg)}"
            )


# Chat input
placeholder = f"Ask a question about `{selected_label}` (e.g., 'top 10 zone routes by volume')"
question = st.chat_input(placeholder)


if question:
    # User bubble
    st.session_state.messages.append({"role": "user", "content": question, "table": selected_table})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"**YOU** · Dataset: `{selected_table}`")
        st.markdown(question)

    # Assistant bubble
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Generating SQL and executing..."):
            out = answer(question, selected_table, limit_rows=1000)
        st.markdown("**Gemini**")
        st.code(out["sql"], language="sql")
        st.write(f"Total rows: {out['total_rows']}")
        st.dataframe(out["df"], use_container_width=True, height=420)
        st.markdown(out["analysis"] or "_(no analysis)_")
        csv_bytes = out["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="query_result.csv", mime="text/csv")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "payload": out, "table": selected_table})
