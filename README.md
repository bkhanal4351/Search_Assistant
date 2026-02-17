# Employee Search Assistant

A smart employee information assistant powered by RAG (Retrieval-Augmented Generation). Ask natural language questions about employees and get accurate answers — no rigid query formats needed.

## How It Works

This app uses a two-stage pipeline to answer questions:

```
User Question
     |
     v
[1. RETRIEVAL] ──> Embedding model (all-MiniLM-L6-v2) converts the question
                   into a vector and finds the top 5 most relevant employee
                   records using cosine similarity.
     |
     v
[2. GENERATION] ──> LLM (Llama 3.3 70B via Groq) reads those records and
                    generates a natural language answer.
     |
     v
Answer displayed in Streamlit UI
```

**Stage 1 — Retrieval:** Each employee record is converted into a readable sentence (e.g., *"Jean Sanchez (ID: 10000) works as Operations Manager in org FIN80697..."*). These sentences are encoded into embeddings and cached. When a user asks a question, it's encoded the same way and compared against all records to find the most relevant matches.

**Stage 2 — Generation:** The top 5 matching records are passed as context to Llama 3.3 70B, which reads them and generates a concise answer. The LLM handles all the intelligence — understanding nicknames, partial names, rephrased questions, and complex queries.

## Project Structure

```
├── app.py              # Main application (RAG pipeline + Streamlit UI)
├── employees.xlsx      # Employee data (2,000 records)
├── .env                # Groq API key (not committed to git)
├── .gitignore          # Excludes .env, cache files, OS junk
├── row_embeddings.pkl  # Auto-generated embedding cache (not committed)
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/bkhanal4351/Search_Assistant.git
cd Search_Assistant
```

### 2. Install dependencies

```bash
pip install streamlit pandas sentence-transformers groq python-dotenv openpyxl
```

### 3. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card required)
3. Create an API key

### 4. Create a `.env` file

```
GROQ_API_KEY=your_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The first run will:
- Download the `all-MiniLM-L6-v2` embedding model (~90MB, one-time)
- Generate and cache embeddings for all employee records

Subsequent runs load from cache and start instantly.

## Example Questions

| Question | What it tests |
|---|---|
| "What is Jean Sanchez's email?" | Direct lookup |
| "Who does Jean report to?" | Supervisor query |
| "List people in the Engineering department" | Department filter |
| "Who works for Kenneth Hughes?" | Reverse supervisor lookup |
| "What's Kenny's email?" | Nickname understanding |
| "Tell me about employee 10000" | ID-based lookup |

## Why No Training or Retraining Is Needed

Neither model in this pipeline requires any training:

**Embedding model (`all-MiniLM-L6-v2`):**
This is a pre-trained sentence transformer. Its job is purely mechanical — convert text into a 384-dimensional vector so we can measure similarity. It already understands English well enough to match "Who is Jean's boss?" to a record containing "Supervisor: Kenneth Hughes". It doesn't need to learn anything specific about your data — it just needs to understand language, which it already does.

**LLM (Llama 3.3 70B):**
This is a 70-billion parameter model pre-trained on vast amounts of text. It already understands natural language, nicknames, context, and can reason over structured data. We don't fine-tune it — we simply provide employee records as context in the prompt and let it generate an answer. This is the core idea behind RAG: instead of training the model on your data, you *retrieve* the relevant data at query time and *feed it to the model*.

**What to tune instead of retraining:**
- **System prompt** (`app.py` lines 62-67) — Adjust instructions to change response style or behavior
- **`TOP_K`** (`app.py` line 82) — Number of records retrieved. Higher = more context for the LLM but slower. Default: 5
- **`temperature`** (`app.py` line 75) — Lower (0.0-0.2) for factual, deterministic answers. Higher (0.5-1.0) for more varied responses. Default: 0.1
- **`max_tokens`** (`app.py` line 76) — Maximum response length. Default: 300

## Swapping Your Own Data

Replace `employees.xlsx` with your own Excel file. Just make sure it has these columns:

```
EMPL_ID, first_name, last_name, email_address, title,
org_code, department, Supervisor_EMPLID, Supervisor_first_name,
Supervisor_last_name, supervisor_email
```

Then delete `row_embeddings.pkl` (so embeddings regenerate) and restart the app.

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` | Semantic search over employee records |
| LLM | Llama 3.3 70B (via Groq) | Natural language answer generation |
| LLM API | Groq (free tier) | Fast inference, 30 requests/min |
| UI | Streamlit | Web interface |
| Data | Pandas + openpyxl | Excel file loading and processing |
