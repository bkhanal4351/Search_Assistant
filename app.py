import os
import hashlib
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Load and clean data ---
df = pd.read_excel("employees.xlsx")
df.rename(columns={
    "EMPL_ID": "empl_id",
    "first_name": "first_name",
    "last_name": "last_name",
    "email_address": "email",
    "title": "title",
    "org_code": "org_code",
    "department": "department",
    "Supervisor_EMPLID": "supervisor_emplid",
    "Supervisor_first_name": "supervisor_first_name",
    "Supervisor_last_name": "supervisor_last_name",
    "supervisor_email": "supervisor_email"
}, inplace=True)


def row_to_text(row):
    return (
        f"{row['first_name']} {row['last_name']} (ID: {row['empl_id']}) works as {row['title']} "
        f"in org {row['org_code']} and department {row.get('department', 'N/A')}. Email: {row['email']}. "
        f"Supervisor: {row['supervisor_first_name']} {row['supervisor_last_name']} "
        f"(ID: {row['supervisor_emplid']}), email: {row['supervisor_email']}."
    )


row_sentences = [row_to_text(row) for _, row in df.iterrows()]

# --- Load embedding model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# --- Load or generate cached embeddings (auto-detects data changes) ---
data_file = "employees.xlsx"
embeddings_file = "row_embeddings.pkl"
data_hash = hashlib.md5(open(data_file, "rb").read()).hexdigest()

rebuild = True
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        cache = pickle.load(f)
    if isinstance(cache, dict) and cache.get("hash") == data_hash:
        row_embeddings = cache["embeddings"]
        rebuild = False

if rebuild:
    row_embeddings = embedding_model.encode(row_sentences, convert_to_tensor=True)
    with open(embeddings_file, "wb") as f:
        pickle.dump({"hash": data_hash, "embeddings": row_embeddings}, f)

# --- Pre-compute summary statistics for aggregate questions ---
def build_summary():
    lines = []
    lines.append(f"Total employees: {len(df)}")

    dept_counts = df['department'].value_counts()
    lines.append("\nEmployees per department:")
    for dept, count in dept_counts.items():
        lines.append(f"  {dept}: {count}")

    org_counts = df['org_code'].value_counts()
    lines.append(f"\nTotal unique org codes: {len(org_counts)}")
    lines.append("\nTop 10 org codes by employee count:")
    for org, count in org_counts.head(10).items():
        lines.append(f"  {org}: {count}")

    title_counts = df['title'].value_counts()
    lines.append("\nEmployees per title:")
    for title, count in title_counts.items():
        lines.append(f"  {title}: {count}")

    sup_counts = (df['supervisor_first_name'] + ' ' + df['supervisor_last_name']).value_counts()
    lines.append("\nTop 10 supervisors by number of direct reports:")
    for sup, count in sup_counts.head(10).items():
        lines.append(f"  {sup}: {count}")

    return "\n".join(lines)


data_summary = build_summary()

# --- Groq LLM (Llama 3.3 70B) ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

AGGREGATE_KEYWORDS = [
    "most", "least", "how many", "count", "total", "average",
    "top", "bottom", "all", "list all", "every", "each",
    "which department", "which org", "biggest", "smallest",
    "percentage", "breakdown", "summary", "statistics"
]


def ask_llm(question, context):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an employee information assistant. "
                    "Answer questions using ONLY the data provided. "
                    "Be concise and direct. If someone uses a nickname or partial name, "
                    "match it to the closest employee name in the records. "
                    "If the answer is not in the data, say 'I could not find that information.'"
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.1,
        max_tokens=300
    )
    return response.choices[0].message.content


# --- RAG: retrieve relevant records then generate answer ---
TOP_K = 5


def get_answer(question):
    q_lower = question.lower()
    is_aggregate = any(kw in q_lower for kw in AGGREGATE_KEYWORDS)

    q_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_indices = scores.topk(k=min(TOP_K, len(row_sentences))).indices.tolist()
    top_score = scores[top_indices[0]].item()

    individual_records = "\n".join(row_sentences[i] for i in top_indices)

    if is_aggregate:
        context = f"Dataset Summary:\n{data_summary}\n\nMatching Records:\n{individual_records}"
    else:
        context = f"Employee Records:\n{individual_records}"

    answer = ask_llm(question, context)
    return answer, top_score


# --- Streamlit UI ---
st.title("Employee Info Assistant")
user_question = st.text_input("Ask a question about an employee:")

if user_question:
    with st.spinner("Thinking..."):
        answer, confidence = get_answer(user_question)

    st.write("**Confidence:**", f"{confidence:.2f}")
    st.markdown(f"**Answer:** {answer}")
