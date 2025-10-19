import os
os.environ["STREAMLIT_ARROW_DATAFRAME_SERIALIZATION"] = "legacy"


import json
import pandas as pd
import streamlit as st
import google.generativeai as genai

# -------------------- Basic Setup --------------------
st.set_page_config(page_title="Lane Imbalance_AI ASSISTANT", page_icon="🤖", layout="wide")
st.title("📊 Lane Imbalance_AI ASSISTANT")

# 1) Fixed dataset path
DATA_PATH = r"C:\Users\wli0001\Downloads\Lane Imbalance Data_R13.csv"

# 2) Paste your (NEW) Gemini API Key here (AI Studio Free Key)
API_KEY = "AIzaSyA-QzLpmqi6HOYiuhDrfglwZg4Tf-6I-js"  # ← replace this with your new key
API_KEY = os.getenv("GEMINI_API_KEY", API_KEY)   # environment variable takes priority if set

if not API_KEY:
    st.error("❌ No Gemini API Key detected. Please set API_KEY in the code or use an environment variable GEMINI_API_KEY.")
    st.stop()

genai.configure(api_key=API_KEY)

# -------------------- Load Dataset (Cached) --------------------
@st.cache_data(show_spinner=True)
def load_dataset(path: str) -> pd.DataFrame:
    """Read the dataset from CSV (you can add Excel/Parquet support if needed)."""
    return pd.read_csv(path, low_memory=False)

try:
    df = load_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# -------------------- Display Data --------------------
#st.caption(f"📁 Data file: {DATA_PATH}")
st.write(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

st.subheader("🔎 Preview (first 1000 rows)")
st.dataframe(df.head(1000))

#st.subheader("🧾 Basic Summary")
#try:
   # st.write(df.describe(include='all', percentiles=[.25, .5, .75]).transpose())
#except Exception:
    #st.write(df.describe().transpose())

# -------------------- Build Dataset Context --------------------
def build_context(sample_rows=5):
    """Prepare a compact summary of the dataset for Gemini to understand context."""
    schema = {c: str(t) for c, t in zip(df.columns, df.dtypes)}
    head_csv = df.head(sample_rows).to_csv(index=False)
    num_desc = df.select_dtypes(include="number").describe().to_csv()

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_info = {}
    for c in cat_cols[:12]:  # limit to avoid overly long prompt
        vc = df[c].astype(str).value_counts(dropna=False).head(10)
        cat_info[c] = vc.to_dict()

    context = {
        "schema": schema,
        "head_rows_csv": head_csv,
        "numeric_summary_csv": num_desc,
        "top_value_counts": cat_info
    }
    return json.dumps(context, ensure_ascii=False)

dataset_context = build_context(sample_rows=5)

# -------------------- Chat Section --------------------
st.divider()
st.subheader("💬 Ask questions about the dataset")

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    st.chat_message("user" if role == "user" else "assistant").write(content)

prompt = st.chat_input("Example: Which column has the highest average?")

if prompt:
    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)

    system_instructions = (
        "You are a data analyst. Answer questions strictly based on the provided dataset summaries and samples. "
        "When calculations are needed, reason step by step and present clear results. "
        "If the question requires data beyond the provided context, infer from summaries or ask for clarification."
    )

    full_prompt = (
        f"{system_instructions}\n\n"
        f"=== DATASET CONTEXT (JSON) ===\n{dataset_context}\n"
        f"=== USER QUESTION ===\n{prompt}\n"
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        resp = model.generate_content(full_prompt)
        answer = resp.text or "(No response)"
    except Exception as e:
        answer = f"⚠️ Model error: {e}"

    st.session_state.messages.append(("assistant", answer))
    st.chat_message("assistant").write(answer)
