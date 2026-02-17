import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Load and clean data ---
# Reads employee records from Excel and standardizes column names
# so the rest of the code can use consistent, lowercase field names.
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


# Converts a single DataFrame row into a readable sentence.
# This sentence format is what gets embedded and later fed to the LLM as context.
def row_to_text(row):
    return (
        f"{row['first_name']} {row['last_name']} (ID: {row['empl_id']}) works as {row['title']} "
        f"in org {row['org_code']} and department {row.get('department', 'N/A')}. Email: {row['email']}. "
        f"Supervisor: {row['supervisor_first_name']} {row['supervisor_last_name']} "
        f"(ID: {row['supervisor_emplid']}), email: {row['supervisor_email']}."
    )


row_sentences = [row_to_text(row) for _, row in df.iterrows()]

# --- Load embedding model and compute embeddings (cached by Streamlit) ---
# all-MiniLM-L6-v2 converts text into 384-dimensional vectors.
# Used to measure semantic similarity between a user's question and employee records.
# @st.cache_resource keeps these in memory across reruns, avoiding redundant computation.
@st.cache_resource
def load_model_and_embeddings(_sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(list(_sentences), convert_to_tensor=True)
    return model, embeddings

embedding_model, row_embeddings = load_model_and_embeddings(tuple(row_sentences))


# --- Pre-compute summary statistics for aggregate questions ---
# Builds a text summary of the entire dataset: employee counts by department,
# org code, title, and supervisor. This is passed to the LLM when the user
# asks aggregate questions (e.g., "which org has the most employees?")
# so it can answer without needing to see every individual record.
def build_summary():
    lines = []
    lines.append(f"Total employees: {len(df)}")

    dept_counts = df['department'].value_counts()
    lines.append("\nEmployees per department:")
    for dept, count in dept_counts.items():
        lines.append(f"  {dept}: {count}")

    org_counts = df['org_code'].value_counts()
    lines.append(f"\nTotal unique org codes: {len(org_counts)}")
    lines.append("\nEmployees per org code:")
    for org, count in org_counts.items():
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
# Initializes the Groq client for calling Llama 3.3 70B.
# API key is loaded from the .env file (never hardcoded).
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Keywords that indicate the user is asking an aggregate/analytical question
# rather than looking up a specific employee. When detected, the LLM receives
# pre-computed summary statistics in addition to individual matching records.
AGGREGATE_KEYWORDS = [
    "most", "least", "how many", "count", "total", "average",
    "top", "bottom", "all", "list all", "every", "each",
    "which department", "which org", "biggest", "smallest",
    "percentage", "breakdown", "summary", "statistics",
    "by org", "by department", "by title", "per org", "per department",
    "per title", "group by", "grouped by", "list of employees",
    "list employees", "how many employees"
]


# Sends the user's question along with retrieved context to the LLM.
# The system prompt instructs the LLM to only use the provided data,
# handle nicknames/partial names, and respond concisely.
def ask_llm(question, context, is_aggregate=False):
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
                    "For broad questions (e.g., 'list employees by org code', "
                    "'how many people per department'), present the counts and "
                    "breakdowns from the Dataset Summary. This IS the answer - "
                    "showing employee counts per group is the correct response "
                    "to these types of questions. "
                    "Only list individual employee names when the user asks about "
                    "specific people, names, or small groups. "
                    "If the question is about a specific person and they are not "
                    "in the data, say 'I could not find that information.'"
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.1,
        max_tokens=1000 if is_aggregate else 300
    )
    return response.choices[0].message.content


# --- Direct dataframe lookup for specific filters ---
# Searches the actual dataframe for names, departments, and org codes
# mentioned in the question. This catches cases that embedding search
# misses, like "who reports to Deepa" (needs all of Deepa's reports,
# not just 5 semantically similar records).
def lookup_records(question):
    q_lower = question.lower()
    matches = pd.DataFrame()

    # Check for supervisor-related questions
    sup_keywords = ["report", "under", "supervis", "manage", "direct report", "team"]
    is_supervisor_query = any(kw in q_lower for kw in sup_keywords)

    # Search by name across employee and supervisor columns
    all_first_names = set(df['first_name'].str.lower().unique()) | set(df['supervisor_first_name'].str.lower().unique())
    all_last_names = set(df['last_name'].str.lower().unique()) | set(df['supervisor_last_name'].str.lower().unique())

    matched_first = [n for n in all_first_names if n in q_lower]
    matched_last = [n for n in all_last_names if n in q_lower]

    if is_supervisor_query and matched_first:
        for name in matched_first:
            sup_matches = df[df['supervisor_first_name'].str.lower() == name]
            if matched_last:
                for lname in matched_last:
                    refined = sup_matches[sup_matches['supervisor_last_name'].str.lower() == lname]
                    if len(refined) > 0:
                        sup_matches = refined
            matches = pd.concat([matches, sup_matches])
    elif matched_first or matched_last:
        for name in matched_first:
            matches = pd.concat([matches, df[df['first_name'].str.lower() == name]])
            matches = pd.concat([matches, df[df['supervisor_first_name'].str.lower() == name]])
        for name in matched_last:
            matches = pd.concat([matches, df[df['last_name'].str.lower() == name]])
            matches = pd.concat([matches, df[df['supervisor_last_name'].str.lower() == name]])

    # Search by department
    all_depts = set(df['department'].str.lower().unique())
    matched_depts = [d for d in all_depts if d in q_lower]
    for dept in matched_depts:
        matches = pd.concat([matches, df[df['department'].str.lower() == dept]])

    # Search by org code
    all_orgs = set(df['org_code'].str.lower().unique())
    matched_orgs = [o for o in all_orgs if o in q_lower]
    for org in matched_orgs:
        matches = pd.concat([matches, df[df['org_code'].str.lower() == org]])

    matches = matches.drop_duplicates(subset='empl_id')
    return matches


# --- RAG pipeline: retrieve relevant records, then generate an answer ---
# 1. Does a direct dataframe lookup for names, departments, and org codes
# 2. Encodes the user's question into a vector for semantic search
# 3. Combines lookup results with top-K embedding matches
# 4. For aggregate questions: includes dataset summary stats
# 5. Passes the context to the LLM to generate a natural language answer
TOP_K = 5


def get_answer(question):
    q_lower = question.lower()
    is_aggregate = any(kw in q_lower for kw in AGGREGATE_KEYWORDS)

    # Direct lookup in the dataframe for exact matches
    lookup_matches = lookup_records(question)
    lookup_texts = [row_to_text(row) for _, row in lookup_matches.iterrows()]

    # Embedding-based semantic search
    q_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, row_embeddings)[0]
    top_indices = scores.topk(k=min(TOP_K, len(row_sentences))).indices.tolist()
    top_score = scores[top_indices[0]].item()

    embedding_records = [row_sentences[i] for i in top_indices]

    # Combine both sources, deduplicating
    all_records = lookup_texts + [r for r in embedding_records if r not in lookup_texts]

    # Cap at 50 records to avoid exceeding LLM context limits.
    # For large result sets, include a count so the LLM knows the full picture.
    record_count = len(all_records)
    if record_count > 50:
        context_records = "\n".join(all_records[:50])
        context_records = f"Showing 50 of {record_count} matching records:\n{context_records}"
    else:
        context_records = "\n".join(all_records)

    if is_aggregate:
        context = f"Dataset Summary:\n{data_summary}\n\nMatching Records ({record_count} found):\n{context_records}"
    else:
        context = f"Employee Records ({record_count} found):\n{context_records}"

    answer = ask_llm(question, context, is_aggregate=is_aggregate)
    return answer, top_score


# --- Streamlit UI ---
# Simple interface: text input for questions, spinner while the LLM thinks,
# then displays the answer with a confidence score (cosine similarity of the
# best matching record - higher means the retrieval was more relevant).
st.title("Employee Info Assistant")
user_question = st.text_input("Ask a question about an employee:")

if user_question:
    with st.spinner("Thinking..."):
        answer, confidence = get_answer(user_question)

    st.write("**Confidence:**", f"{confidence:.2f}")
    st.markdown(f"**Answer:** {answer}")
