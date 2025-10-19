# nl2sql_streamlit_db_chat.py
import os
import json
import re
import math
import pandas as pd
import duckdb
import streamlit as st


DB_PATH = "/tmp/lane.duckdb"

@st.cache_resource
def get_duckdb_connection():
    url = st.secrets.get("DUCKDB_URL") or os.getenv("DUCKDB_URL")
    if not url:
        st.error("❌ Missing DUCKDB_URL (please set it in Streamlit Secrets)")
        st.stop()

    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 1024:
        st.info("⬇️ Downloading lane.duckdb ...")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(DB_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
        st.success("✅ Database downloaded successfully")

    con = duckdb.connect(DB_PATH, read_only=True)
    return con

con = get_duckdb_connection()





# Optional: Plotly for charts (fallback to st.bar_chart if missing)
HAS_PLOTLY = True
try:
    import plotly.express as px
except Exception:
    HAS_PLOTLY = False

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
    ("Both datasets (smart join)", "__both__"),
]
DEFAULT_TABLE = "lane_imbalance_clean"

SUGGESTIONS = {
    "lane_imbalance_clean": [
        ("Top 10 carriers by Load Count",
         "Top 10 carriers by Load Count; show columns Carrier Name and Load Count; order by Load Count desc"),
        ("Bar chart: Top 5 carriers by Load Count",
         "Bar chart of top 5 carriers by Load Count"),
        ("Linehaul Cost by Province (2024)",
         "Sum of Linehaul Cost by Province for Year = '2024'; sort desc; show top 10; chart"),
        ("Avg Distance by Carrier",
         "Average Distance (Miles) by Carrier Name; show top 10; order by average distance desc; chart"),
        ("Top 10 Zone Routes by Volume",
         "Top 10 Zone Routes by Volume (Cubic Feet); include Zone Route and total volume; chart"),
        ("Help", "help"),
    ],
    "target_tonnage_clean": [
        ("Top 10 vendors by PO Count",
         "Top 10 Vendor by PO Count; order by PO Count desc"),
        ("Revenue by DC",
         "Sum of Estimated Revenue by DC; show top 10; chart"),
        ("Profit by Customer",
         "Sum of Estimated Profit by Customer ID; show top 10; chart"),
        ("DED% distribution",
         "Show average DED% by DC; order desc; chart"),
        ("R13 Province Score Top 10",
         "Top 10 by R13 Province Score with Vendor and Customer ID; order desc"),
        ("Help", "help"),
    ],
    "__both__": [
        ("Compare: Linehaul Cost vs Revenue by Province (2024)",
         "Compare total Linehaul Cost from lane_imbalance table and total Estimated Revenue from target_tonnage table by Province for Year='2024'; show both metrics side-by-side; order by revenue desc; chart"),
        ("Top DC: Profit (target) and Loads (lane)",
         "Show DC with highest Estimated Profit (target_tonnage) and match with total Load Count (lane_imbalance); join on DC if exists; show top 10; chart"),
        ("Customer overlap: Profit vs Distance",
         "For customers that exist in target_tonnage (Customer ID) and in lane_imbalance (if Customer ID exists), compare total Estimated Profit and average Distance (Miles); top 10 by profit; chart"),
        ("Vendor/Carrier blend by Province",
         "Aggregate Linehaul Cost by Province from lane_imbalance and Estimated Revenue by Province from target_tonnage; present both; show 10 highest revenue provinces; chart"),
        ("Help", "help"),
    ],
}

# ===================== Streamlit Config =====================
def _set_option_safe(key: str, value):
    try:
        st.set_option(key, value)
    except Exception:
        pass
_set_option_safe("server.fileWatcherType", "none")

st.set_page_config(page_title="Lane Imbalance Data_AI Bot", layout="wide")

# -------- Dark Purple theme (stronger contrast) + rounded cards --------
st.markdown("""
<style>
:root{
  --pc-purple:#5B2DFF;
  --pc-purple-d:#4a22e0;
  --pc-bg:#0c0a15;           /* darker */
  --pc-card:#141028;         /* darker */
  --pc-soft:#1b1536;         /* darker */
  --pc-text:#FFFFFF;         /* brighter */
  --pc-muted:#D9D3FF;        /* brighter muted */
  --pc-border:rgba(255,255,255,0.18); /* stronger border */
}

/* page background */
section.main > div {padding-top: 0.5rem;}
html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, var(--pc-bg) 0%, #090713 100%);
}
h1, h2, h3, h4, h5, h6, p, code, span, div, label{ color: var(--pc-text) !important; }

/* top toolbar / hamburger menu */
[data-testid="stToolbar"]{
  background: linear-gradient(90deg, var(--pc-soft), var(--pc-card)) !important;
  border-bottom: 1px solid var(--pc-border);
}
[data-testid="stToolbar"] *{ color: var(--pc-text) !important; }

/* header title card */
.app-header-card{
  background: linear-gradient(135deg, var(--pc-soft), var(--pc-card));
  border: 1px solid var(--pc-border);
  box-shadow: 0 10px 28px rgba(0,0,0,0.45);
  border-radius: 20px; padding: 14px 18px; margin-bottom: 10px;
}

/* sidebar */
section[data-testid="stSidebar"]{
  background: rgba(20,16,40,0.95) !important;
  border-right: 1px solid var(--pc-border);
}
section[data-testid="stSidebar"] * { color: var(--pc-text) !important; }
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stTextInput{
  background: var(--pc-card); border-radius: 14px; border:1px solid var(--pc-border);
}

/* buttons -> pill cards */
.stButton>button{
  background: var(--pc-purple);
  color: white; border:1px solid #3a1cc6; border-radius: 16px !important;
  padding: 12px 16px; font-weight: 700;
  box-shadow: 0 8px 18px rgba(91,45,255,0.45);
  transition: all .15s ease;
}
.stButton>button:hover{ background: var(--pc-purple-d); transform: translateY(-1px); }
.stButton>button:active{ transform: translateY(0px) scale(.99); }

/* expander */
details[data-testid="stExpander"]{
  background: var(--pc-card) !important; border-radius: 16px !important;
  border: 1px solid var(--pc-border);
}
summary p{ color: var(--pc-muted) !important; }

/* dataframe */
div[data-testid="stDataFrame"]{
  background: var(--pc-card) !important; 
  border-radius: 16px; border:1px solid var(--pc-border);
  box-shadow: 0 8px 18px rgba(0,0,0,.38);
}

/* chat bubbles */
[data-testid="stChatMessage"]{
  border-radius: 16px; padding: 8px 12px;
  background: var(--pc-card) !important;
  border: 1px solid var(--pc-border);
  box-shadow: 0 8px 18px rgba(0,0,0,0.38);
}
[data-testid="stChatMessage"]:has(img[alt="🧑‍💻"]){
  background: rgba(91,45,255,0.16) !important;
  border-color: rgba(91,45,255,0.50) !important;
}
[data-testid="stChatMessage"]:has(img[alt="✨"]){
  background: rgba(27,21,54,0.92) !important;
}

/* code blocks */
pre, code{
  background: #0f0b22 !important; color: #f0ecff !important; 
  border-radius: 12px !important; border: 1px solid var(--pc-border);
}

/* quick actions subtitle & divider */
.qa-subtitle{ font-weight:800; color: var(--pc-muted); letter-spacing:.2px; margin: 6px 0 12px 2px; }
.qa-divider{ height:1px; background: var(--pc-border); margin: 10px 0 16px 0; border-radius: 999px;}

/* plotly wrapper */
.js-plotly-plot, .plot-container{
  border-radius: 16px; overflow: hidden;
  border: 1px solid var(--pc-border);
  box-shadow: 0 8px 18px rgba(0,0,0,.38);
}

/* download button */
[data-testid="baseButton-secondary"]{
  color:#fff !important; background: var(--pc-soft) !important; border-radius:14px !important; 
  border:1px solid var(--pc-border) !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="app-header-card"><h1>⚙️📈 Lane Imbalance Data_AI Bot</h1>'
    '<p style="margin:0;color:#D9D3FF">💬 Ask questions about the dataset</p></div>',
    unsafe_allow_html=True
)

# ===================== DB Connection =====================
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

    def _shape(table_name: str):
        try:
            rows = int(con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0])
            cols = len(con.execute(f"DESCRIBE SELECT * FROM {table_name} LIMIT 0").df())
            return rows, cols
        except Exception:
            return 0, 0

    if selected_table == "__both__":
        r1, c1 = _shape("lane_imbalance_clean")
        r2, c2 = _shape("target_tonnage_clean")
        st.markdown(f"📐 **Lane Imbalance:** {r1:,} × {c1}  \n📐 **Target Tonnage:** {r2:,} × {c2}")
    else:
        rows, cols = _shape(selected_table)
        st.markdown(f"📐 **Shape:** {rows:,} rows × {cols} columns")

    st.divider()
    st.markdown("**Gemini model**"); st.code(MODEL)
    st.markdown("**API key source**"); st.write("Hard-coded" if API_KEY else "Environment variable or missing")
    if not HAS_GENAI:
        st.warning("google-generativeai not installed; only manual SQL will work.", icon="⚠️")
    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None); st.experimental_rerun()

# ===================== Helpers (Schema/SQL) =====================
@st.cache_data(show_spinner=False)
def _schema_for_prompt(table: str):
    schema_df = con.execute(f"DESCRIBE SELECT * FROM {table} LIMIT 0").df()
    schema = {row["column_name"]: row["column_type"] for _, row in schema_df.iterrows()}
    head_df = con.execute(f"SELECT * FROM {table} LIMIT 6").df()
    return schema, head_df

@st.cache_data(show_spinner=False)
def _schemas_for_both():
    s1, h1 = _schema_for_prompt("lane_imbalance_clean")
    s2, h2 = _schema_for_prompt("target_tonnage_clean")
    return (s1, h1), (s2, h2)

def clean_sql(sql: str) -> str:
    s = (sql or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m: s = m.group(1)
    s = re.sub(r"^`+|`+$", "", s).strip()
    return s

def try_execute_sql(sql: str, limit_rows: int = 1000):
    df = con.execute(sql).df()
    head = df.head(limit_rows)
    return {"rows": head.to_dict(orient="records"), "columns": list(head.columns), "total_rows": len(df), "df": df}

def plan_sql(question: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        return "SELECT 'LLM disabled. Please enable Gemini API key.' AS note;" if table == "__both__" else f"SELECT * FROM {table} LIMIT 50;"
    if table == "__both__":
        (s1, h1), (s2, h2) = _schemas_for_both()
        sys = f"""
You are a senior data analyst. Generate **valid DuckDB SQL** to answer the question using one or both tables:
- lane_imbalance_clean AS li
- target_tonnage_clean AS tt

Return only SQL (you may use ```sql fences). SELECT only. Use aliases li, tt.

SCHEMAS:
[li] {s1}
Sample li (CSV): {h1.to_csv(index=False)}

[tt] {s2}
Sample tt (CSV): {h2.to_csv(index=False)}

JOIN RULES:
- Join only on shared, semantically matched columns (Province, DC, Customer ID, Year/Period/Week if both exist).
- If unsure, aggregate per-table and then join on the safest common dimension (Province / DC / Customer ID / Year).
- No invented columns; avoid ambiguous joins.
- For top-N, use ORDER BY ... DESC LIMIT N and clear metric aliases (snake_case).
"""
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content([sys, f"Question: {question}\nWrite the SQL now."])
        return (resp.text or "").strip()
    else:
        schema, head_df = _schema_for_prompt(table)
        sys = f"""
You are a senior data analyst. Generate **valid DuckDB SQL** to answer the question.
Use only the table `{table}`. SELECT only. Return only SQL.

Schema:
{json.dumps(schema, indent=2)}

Sample rows (CSV):
{head_df.to_csv(index=False)}

Guidelines:
- Use ORDER BY + LIMIT for top/bottom; GROUP BY with SUM/AVG/COUNT when needed.
- Use exact column names; avoid ambiguity.
"""
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content([sys, f"Question: {question}\nWrite the SQL now."])
        return (resp.text or "").strip()

def repair_sql(question: str, prev_sql: str, error_msg: str, table: str) -> str:
    if not (HAS_GENAI and API_KEY):
        return f"SELECT 'SQL failed: {error_msg}' AS error"
    if table == "__both__":
        (s1, h1), (s2, h2) = _schemas_for_both()
        prompt = f"""
Fix SQL for DuckDB.

Question:
{question}

Failed SQL:
{prev_sql}

Error:
{error_msg}

[li schema] {s1}
Sample li: {h1.to_csv(index=False)}

[tt schema] {s2}
Sample tt: {h2.to_csv(index=False)}

Rules:
- Use li/tt aliases only; join on safe common keys; aggregate separately if needed.
Return corrected SQL only.
"""
    else:
        schema, head_df = _schema_for_prompt(table)
        prompt = f"""
Fix SQL for DuckDB.

Question:
{question}

Failed SQL:
{prev_sql}

Error:
{error_msg}

Schema:
{json.dumps(schema, indent=2)}

Sample rows:
{head_df.to_csv(index=False)}

Return the corrected SQL only.
"""
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

def summarize_insights(question: str, used_sql: str, result_df: pd.DataFrame) -> str:
    if HAS_GENAI and API_KEY:
        model = genai.GenerativeModel(MODEL)
        preview_csv = result_df.head(20).to_csv(index=False)
        prompt = f"""
You are a business analyst. Read the question, SQL, and the result preview (CSV).
Write 3-6 concise, actionable INSIGHTS for decision-makers. No SQL explanation.

Question:
{question}

SQL:
{used_sql}

Result preview (up to 20 rows):
{preview_csv}
"""
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    if result_df.empty: return "- No rows returned."
    msgs, num_cols = [], result_df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        m = num_cols[0]; top_row = result_df.iloc[0]
        msgs.append(f"- Peak value on '{m}' is {top_row[m]:,} at row 1.")
        msgs.append(f"- Mean of '{m}' on the sample is {result_df[m].mean():,.2f}.")
        msgs.append(f"- Min of '{m}' on the sample is {result_df[m].min():,}.")
    else:
        msgs.append("- Results available; no numeric metrics detected for summary.")
    return "\n".join(msgs)

# ===================== Helpers (UI/Logic) =====================
def is_topn_query(text: str):
    if not text: return (False, None)
    m = re.search(r'(?:top\s*-?\s*(\d+))|(?:front\s*(\d+))|(?:first\s*(\d+))|(?:前\s*(\d+))', text, flags=re.IGNORECASE)
    if m:
        n = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        try: return (True, int(n))
        except Exception: return (True, None)
    return (False, None)

def calc_table_height(num_rows: int, header_px: int = 38, row_px: int = 36,
                      min_px: int = 110, max_px: int = 420) -> int:
    h = header_px + max(1, num_rows) * row_px + 8
    return max(min(h, max_px), min_px)

CHAT_PATTERNS = [
    r"^hi\b", r"^hello\b", r"^hey\b",
    r"\bthanks?\b", r"\bthank you\b",
    r"\bgot it\b", r"\bok\b", r"\bokay\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    r"\bbye\b", r"\bsee you\b",
]
HELP_PATTERNS = [r"\bhelp\b", r"how to use", r"usage", r"what can you do"]

def detect_intent(text: str):
    if not text: return "data"
    t = text.strip().lower()
    if any(re.search(p, t, flags=re.IGNORECASE) for p in HELP_PATTERNS): return "help"
    if any(re.search(p, t, flags=re.IGNORECASE) for p in CHAT_PATTERNS): return "chat"
    data_keywords = [
        "top", "sum", "avg", "average", "min", "max", "group by", "order by", "count",
        "trend", "load", "weight", "volume", "cost", "distance", "score", "revenue", "profit",
        "chart", "plot", "graph", "visual", "line", "bar"
    ]
    if any(k in t for k in data_keywords) or "?" in t: return "data"
    return "chat" if len(t) <= 12 else "data"

def wants_chart(text: str) -> bool:
    if not text: return False
    t = text.lower()
    return any(k in t for k in ["chart", "plot", "graph", "visual", "line", "bar", "趋势", "图"])

def pick_chart_axes(df: pd.DataFrame):
    cols = list(df.columns)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    val = num_cols[0] if num_cols else None
    cat = None
    for c in cols:
        if c.lower() == "rank": continue
        if c not in num_cols: cat = c; break
    kind = "bar"
    if cat and re.search(r"(date|day|week|month|period|year|time)", cat, flags=re.IGNORECASE): kind = "line"
    return cat, val, kind

def render_chart(df: pd.DataFrame, question: str):
    """Plotly chart with explicit dark template and readable colors."""
    if df.empty: return
    cat, val, kind = pick_chart_axes(df)
    if not cat or not val: return
    title = f"{kind.title()} of {val} by {cat}"

    if HAS_PLOTLY:
        if kind == "bar":
            fig = px.bar(df, x=cat, y=val, title=title)
        else:
            fig = px.line(df, x=cat, y=val, title=title, markers=True)

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
            paper_bgcolor="#141028",
            plot_bgcolor="#141028",
            font=dict(color="#FFFFFF"),
            xaxis=dict(
                title=cat, gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.25)"
            ),
            yaxis=dict(
                title=val, gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.25)"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        try:
            st.bar_chart(df.set_index(cat)[val])
        except Exception:
            pass

def answer(question: str, table: str, limit_rows: int = 1000):
    sql1 = clean_sql(plan_sql(question, table))
    try:
        result = try_execute_sql(sql1, limit_rows=limit_rows); used_sql = sql1
    except Exception as e:
        sql2 = clean_sql(repair_sql(question, sql1, str(e), table))
        result = try_execute_sql(sql2, limit_rows=limit_rows); used_sql = sql2
    insights = summarize_insights(question, used_sql, result["df"])
    return {"sql": used_sql, **result, "insights": insights}

# ---- Quick actions panel ----
def render_quick_actions(selected_table_key: str):
    st.markdown('<div class="qa-subtitle">Quick actions</div>', unsafe_allow_html=True)
    suggs = SUGGESTIONS.get(selected_table_key, [])
    if not suggs:
        st.markdown('<div class="qa-divider"></div>', unsafe_allow_html=True); return None
    cols = st.columns(2); fired = None
    for i, (label, prompt) in enumerate(suggs[:4]):
        if cols[i % 2].button(label, use_container_width=True, key=f"qa_{selected_table_key}_{i}"):
            fired = prompt
    with st.expander("More options"):
        cols2 = st.columns(2)
        for j, (label, prompt) in enumerate(suggs[4:], start=4):
            if cols2[j % 2].button(label, use_container_width=True, key=f"qa_{selected_table_key}_{j}"):
                fired = prompt
    st.markdown('<div class="qa-divider"></div>', unsafe_allow_html=True)
    return fired

# ===================== Chat State =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===================== Quick Actions =====================
preset_question = render_quick_actions(selected_table)

# ===================== Render chat history =====================
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**YOU** · Dataset: `{msg.get('display', msg.get('table',''))}`")
            st.markdown(msg["content"])
    else:
        if msg.get("kind") in ("chat", "help"):
            with st.chat_message("assistant", avatar="✨"):
                st.markdown("**Gemini**"); st.markdown(msg["text"])
            continue
        prev_q = ""
        for j in range(i - 1, -1, -1):
            if st.session_state.messages[j]["role"] == "user":
                prev_q = st.session_state.messages[j]["content"]; break
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**"); st.code(msg["payload"]["sql"], language="sql")
            render_df = msg["payload"]["df"].copy()
            topn, n = is_topn_query(prev_q)
            if topn: render_df.insert(0, "Rank", range(1, len(render_df) + 1))
            height = calc_table_height(len(render_df))
            st.dataframe(render_df, use_container_width=True, height=height, hide_index=True)
            st.markdown("**Insights**"); st.markdown(msg["payload"]["insights"] or "_(no insights)_")
            if wants_chart(prev_q): render_chart(render_df, prev_q)
            csv_bytes = render_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="query_result.csv", mime="text/csv",
                               key=f"dl_{id(msg)}")

# ===================== Chat Input =====================
label_for_placeholder = selected_label if selected_table != "__both__" else "both datasets"
placeholder = f"Ask a question about `{label_for_placeholder}` (e.g., 'compare revenue and linehaul by province (2024)')"
question = preset_question or st.chat_input(placeholder)

if question:
    display_label = selected_label if selected_table != "__both__" else "Both datasets"
    st.session_state.messages.append(
        {"role": "user", "content": question, "table": selected_table, "display": display_label}
    )
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"**YOU** · Dataset: `{display_label}`"); st.markdown(question)

    intent = detect_intent(question)

    if intent == "chat":
        reply = f"You're welcome! Want me to analyze something from **{display_label}**? e.g., 'Compare revenue and linehaul by province (2024)'."
        st.session_state.messages.append({"role": "assistant", "kind": "chat", "text": reply})
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**"); st.markdown(reply)

    elif intent == "help":
        help_text = (
            f"I can translate natural language into SQL and run it on **{display_label}**.\n\n"
            "Try:\n"
            "- Compare **Estimated Revenue** (target) vs **Linehaul Cost** (lane) by Province (2024)\n"
            "- Top DC by **Profit** (target) and their total **Load Count** (lane)\n"
            "- Customer-level **Profit vs Avg Distance** comparison\n"
            "- Bar chart of top 5 carriers by **Load Count** (lane)\n"
        )
        st.session_state.messages.append({"role": "assistant", "kind": "help", "text": help_text})
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("**Gemini**"); st.markdown(help_text)

    else:
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Generating SQL and executing..."):
                out = answer(question, selected_table, limit_rows=1000)
            st.markdown("**Gemini**"); st.code(out["sql"], language="sql")
            render_df = out["df"].copy()
            topn, n = is_topn_query(question)
            if topn: render_df.insert(0, "Rank", range(1, len(render_df) + 1))
            height = calc_table_height(len(render_df))
            st.dataframe(render_df, use_container_width=True, height=height, hide_index=True)
            st.markdown("**Insights**"); st.markdown(out["insights"] or "_(no insights)_")
            if wants_chart(question): render_chart(render_df, question)
            csv_bytes = render_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="query_result.csv", mime="text/csv")
        st.session_state.messages.append(
            {"role": "assistant", "payload": out, "table": selected_table, "display": display_label}
        )
