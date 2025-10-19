# nl2sql_streamlit_db_chat.py
import os
import json
import re
import math
import duckdb
import streamlit as st

# ===================== Gemini Config =====================
HAS_GENAI = True
try:
    import google.generativeai as genai
except Exception:
    HAS_GENAI = False

MODEL = "models/gemini-2.0-flash-lite"
API_KEY = "AIzaSyA-QzLpmqi6HOYiuhDrfglwZg4Tf-6I-js".strip()  # your key
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if HAS_GENAI and API_KEY:
    genai.configure(api_key=API_KEY)

# ===================== App Config =====================
DB_PATH = r"C:\Users\wli0001\Lane Imbalance\lane.duckdb"
TABLE_OPTIONS = [
    ("Lane Imbalance", "lane_imbalance_clean"),
    ("Target Tonnage", "target_tonnage_clean"),
]
DEFAULT_TABLE = "lane_imbalance_clean"

# ===================== Streamlit Config =====================
def _set_option_safe(key: str, value):
    try:
        st.set_option(key, value)
    except Exception:
        pass

_set_option_safe("server.fileWatcherType", "none")

st.set_page_config(page_title="Lane Imbalance Data_AI Bot", layout="wide")
st.title("⚙️📈 Lane Imbalance Data_AI Bot")
st.caption("💬 Ask questions about the dataset")

# ===================== Database Connection =====================
@st.cache_resource(show_spinner=False)
def get_con(db_path: str):
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

    # Show dataset shape
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

    st.markdown(f"📐 **Shape:** {rows:,} rows × {cols} columns")

    st.divider()
    st.markdown("**Gemini model**")
    st.code(MODEL)
    st.markdown("**API key source**")
    st.write("Hard-coded" if API_KEY else "Environment variable or missing")
    if not HAS_GENAI:
        st.warning("google-generativeai not installed; only manual SQL will work.", icon="⚠️")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.experimental_rerun()

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def _schema_for_prompt(table: str):
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

def plan_sql(question: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"SELECT * FROM {table} LIMIT 50;"
    schema, head_df = _schema_for_prompt(table)
    system_prompt = f"""
You are a senior data analyst. Generate **valid DuckDB SQL** to answer the question.
Use only the table `{table}`. No DDL, CREATE, UPDATE, DELETE — only SELECT.
Return only SQL (optionally inside ```sql code fences).

Schema:
{json.dumps(schema, indent=2)}

Sample rows (CSV):
{head_df.to_csv(index=False)}

Guidelines:
- Use ORDER BY and LIMIT for top/bottom queries.
- Use GROUP BY for aggregations (SUM, AVG, COUNT).
- Use exact column names from schema.
"""
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content([system_prompt, f"Question: {question}\nWrite the SQL now."])
    return (resp.text or "").strip()

def repair_sql(question: str, prev_sql: str, error_msg: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"SELECT * FROM {table} LIMIT 50;"
    schema, head_df = _schema_for_prompt(table)
    repair_prompt = f"""
The previous SQL failed to run. Fix it.

Question:
{question}

Previous SQL:
{prev_sql}

DuckDB Error:
{error_msg}

Schema:
{json.dumps(schema, indent=2)}

Sample rows (CSV):
{head_df.to_csv(index=False)}

Rules:
- Return corrected SQL only.
- Use only existing columns.
"""
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content(repair_prompt)
    return (resp.text or "").strip()

def summarize(question: str, used_sql: str, result: dict) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"- Returned {result['total_rows']} rows\n- SQL: {used_sql}"
    resp = genai.GenerativeModel(MODEL).generate_content(f"""
Question: {question}

SQL used:
{used_sql}

First {min(10, len(result['df']))} rows shown; total_rows={result['total_rows']}.
Write concise bullet points summarizing the answer.
""")
    return (resp.text or "").strip()

# ---- Utility: Compact table height and Rank ----
def is_topn_query(text: str):
    if not text:
        return (False, None)
    m = re.search(r'(?:top\s*-?\s*(\d+))|(?:front\s*(\d+))|(?:first\s*(\d+))|(?:前\s*(\d+))', text, flags=re.IGNORECASE)
    if m:
        n = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        try:
            return (True, int(n))
        except Exception:
            return (True, None)
    return (False, None)

def calc_table_height(num_rows: int, header_px: int = 38, row_px: int = 36,
                      min_px: int = 110, max_px: int = 420) -> int:
    h = header_px + max(1, num_rows) * row_px + 8
    return max(min(h, max_px), min_px)

# ---- Intent Detection ----
CHAT_PATTERNS = [
    r"^hi\b", r"^hello\b", r"^hey\b",
    r"\bthanks?\b", r"\bthank you\b",
    r"\bgot it\b", r"\bok\b", r"\bokay\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    r"\bbye\b", r"\bsee you\b",
]
HELP_PATTERNS = [r"\bhelp\b", r"how to use", r"usage", r"what can you do"]

def detect_intent(text: str):
    if not text:
        return "data"
    t = text.strip().lower()
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HELP_PATTERNS):
        return "help"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in CHAT_PATTERNS):
        return "chat"
    data_keywords = [
        "top", "sum", "avg", "average", "min", "max", "group by", "order by", "count",
        "trend", "load", "weight", "volume", "cost", "distance", "score", "revenue", "profit"
    ]
    if any(k in t for k in data_keywords) or "?" in t:
        return "data"
    return "chat" if len(t) <= 12 else "data"

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

# ===================== Chat State =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**YOU** · Dataset: `{msg.get('display', msg.get('table',''))}`")
            st.markdown(msg["content"])
    else:
        if msg.get("kind") in ("chat", "help"):
            with st.chat_message("assistant", avatar="✨"):
                st.markdown("**Gemini**")
                st.markdown(msg["text"])
            continue

        prev_q = ""
        for j in range(i - 1, -1, -1):
            if st.session_state.messages[j]["role"] == "user":
                prev_q = st.session_state.messages[j]["content"]
                break

        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**")
            st.code(msg["payload"]["sql"], language="sql")

            render_df = msg["payload"]["df"].copy()
            topn, n = is_topn_query(prev_q)
            if topn:
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

# ===================== Chat Input =====================
placeholder = f"Ask a question about `{selected_label}` (e.g., 'top 10 zone routes by volume')"
question = st.chat_input(placeholder)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "table": selected_table, "display": selected_label}
    )
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"**YOU** · Dataset: `{selected_label}`")
        st.markdown(question)

    intent = detect_intent(question)

    if intent == "chat":
        reply = f"You're welcome! Would you like me to analyze something from **{selected_label}**? (e.g., 'Top 10 carriers by Load Count')"
        st.session_state.messages.append({"role": "assistant", "kind": "chat", "text": reply})
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**")
            st.markdown(reply)

    elif intent == "help":
        help_text = (
            f"I can translate your natural language questions into SQL and run them on **{selected_label}**.\n\n"
            "Try asking things like:\n"
            "- Top 5 carriers by **Load Count**\n"
            "- Sum of **Linehaul Cost** by **Province** in 2024\n"
            "- Average **Distance (Miles)** by **Carrier Name**\n"
            "- Top 10 **Zone Routes** by **Volume (Cubic Feet)**\n"
        )
        st.session_state.messages.append({"role": "assistant", "kind": "help", "text": help_text})
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**")
            st.markdown(help_text)

    else:
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Generating SQL and executing..."):
                out = answer(question, selected_table, limit_rows=1000)

            st.markdown("**Gemini**")
            st.code(out["sql"], language="sql")

            render_df = out["df"].copy()
            topn, n = is_topn_query(question)
            if topn:
                render_df.insert(0, "Rank", range(1, len(render_df) + 1))

            height = calc_table_height(len(render_df))
            st.write(f"Total rows: {out['total_rows']}")
            st.dataframe(render_df, use_container_width=True, height=height, hide_index=True)

            st.markdown(out["analysis"] or "_(no analysis)_")
            csv_bytes = render_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="query_result.csv", mime="text/csv")

        st.session_state.messages.append(
            {"role": "assistant", "payload": out, "table": selected_table, "display": selected_label}
        )
