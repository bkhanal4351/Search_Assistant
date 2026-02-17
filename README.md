# Employee Search Assistant

A smart employee information assistant powered by RAG (Retrieval-Augmented Generation). Ask natural language questions about employees and get accurate answers without rigid query formats.

## How It Works

This app uses a three-stage pipeline to answer questions:

```
User Question
     |
     v
[1. DIRECT LOOKUP] ──> Searches the dataframe for names, departments,
                       and org codes mentioned in the question.
     |
     v
[2. SEMANTIC SEARCH] ──> Embedding model (all-MiniLM-L6-v2) converts the
                         question into a vector and finds the top 5 most
                         relevant records using cosine similarity (kNN).
     |
     v
[3. GENERATION] ──> LLM (Llama 3.3 70B via Groq) reads the combined
                    results and generates a natural language answer.
     |
     v
Answer displayed in Streamlit UI
```

**Stage 1 - Direct Lookup:** Before any embedding search, the app checks the question for recognizable names, departments, and org codes. If the question is about a supervisor (e.g., "who reports to Deepa"), it filters the dataframe for all employees under that supervisor. This catches cases where embedding search alone would miss records, like counting all direct reports for a specific manager.

**Stage 2 - Semantic Search:** Each employee record is converted into a readable sentence (e.g., *"Jean Sanchez (ID: 10000) works as Operations Manager in org FIN80697..."*). These sentences are encoded into embeddings and cached. When a user asks a question, it's encoded the same way and compared against all records via cosine similarity to find the top 5 most relevant matches. Results from both stages are combined and deduplicated.

**Stage 3 - Generation:** The combined records are passed as context to Llama 3.3 70B, which reads them and generates a concise answer. The LLM handles nickname understanding, partial name matching, rephrased questions, and complex queries. For aggregate questions (e.g., "which department has the most employees"), pre-computed summary statistics are also included in the context.

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

## Setup for Windows Users

### Prerequisites

1. Install **Python 3.9+** from [python.org](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.
2. Open **Command Prompt** or **PowerShell**.

### Install dependencies

```cmd
pip install streamlit pandas sentence-transformers groq python-dotenv openpyxl
```

If `pip` is not recognized, try:

```cmd
python -m pip install streamlit pandas sentence-transformers groq python-dotenv openpyxl
```

### Create the `.env` file

In the project folder, create a file named `.env` (no filename, just the extension). You can do this in PowerShell:

```powershell
echo GROQ_API_KEY=your_api_key_here > .env
```

Or in Command Prompt:

```cmd
echo GROQ_API_KEY=your_api_key_here > .env
```

Alternatively, create it manually in any text editor and save it as `.env` (make sure it's not saved as `.env.txt` - turn on "Show file extensions" in File Explorer to verify).

### Run the app

```cmd
streamlit run app.py
```

If that doesn't work:

```cmd
python -m streamlit run app.py
```

### Common Windows Issues

| Issue | Fix |
|---|---|
| `pip` not recognized | Use `python -m pip install ...` instead |
| `streamlit` not recognized | Use `python -m streamlit run app.py` instead |
| `.env` file saved as `.env.txt` | Turn on "Show file extensions" in File Explorer settings and rename |
| PyTorch installation fails | Run `pip install torch` separately before installing sentence-transformers |
| Permission errors | Run Command Prompt as Administrator, or use `pip install --user ...` |

## Embedding Cache (Smart Auto-Rebuild)

The app caches employee record embeddings in `row_embeddings.pkl` to avoid re-encoding on every startup. The cache is **self-managing**:

**How it works:**
1. On startup, the app computes an MD5 hash of `employees.xlsx`
2. It compares this hash against the one stored in the cache file
3. **Hash matches** - loads cached embeddings instantly (fast startup)
4. **Hash differs** (data was modified) - automatically regenerates embeddings and saves a new cache
5. **No cache exists** (first run) - generates embeddings from scratch

**You never need to manually delete the cache.** Swap `employees.xlsx` with new data, restart the app, and it handles everything.

```
Startup Flow:
  employees.xlsx ──> MD5 hash ──> Compare with cached hash
                                       |
                          ┌─────────────┴─────────────┐
                          |                           |
                      Hash matches               Hash differs
                          |                           |
                    Load from cache          Re-encode all records
                     (instant)              Save new cache + hash
```

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
This is a pre-trained sentence transformer. Its job is purely mechanical: convert text into a 384-dimensional vector so we can measure similarity. It already understands English well enough to match "Who is Jean's boss?" to a record containing "Supervisor: Kenneth Hughes". It doesn't need to learn anything specific about your data. It just needs to understand language, which it already does.

**LLM (Llama 3.3 70B):**
This is a 70-billion parameter model pre-trained on vast amounts of text. It already understands natural language, nicknames, context, and can reason over structured data. We don't fine-tune it. We simply provide employee records as context in the prompt and let it generate an answer. This is the core idea behind RAG: instead of training the model on your data, you *retrieve* the relevant data at query time and *feed it to the model*.

**What to tune instead of retraining:**

- **System prompt** (the `"role": "system"` message inside `ask_llm()`) - Adjust instructions to change response style or behavior
- **`TOP_K`** (defined before `get_answer()`) - Number of records retrieved via semantic search. Higher = more context but slower. Default: 5
- **`temperature`** (inside `ask_llm()`) - Lower (0.0-0.2) for factual, deterministic answers. Higher (0.5-1.0) for more varied responses. Default: 0.1
- **`max_tokens`** (inside `ask_llm()`) - Maximum response length. Default: 300

## Swapping Your Own Data

Replace `employees.xlsx` with your own Excel file. Just make sure it has these columns:

```
EMPL_ID, first_name, last_name, email_address, title,
org_code, department, Supervisor_EMPLID, Supervisor_first_name,
Supervisor_last_name, supervisor_email
```

The app automatically detects data changes and regenerates embeddings. No manual steps needed.

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` | Semantic search over employee records |
| LLM | Llama 3.3 70B (via Groq) | Natural language answer generation |
| LLM API | Groq (free tier) | Fast inference, 30 requests/min |
| UI | Streamlit | Web interface |
| Data | Pandas + openpyxl | Excel file loading and processing |

## Scaling to Production

This section covers what you need to move from a local laptop setup to a shared deployment serving real users.

### Deploying on a Server or VDI

To run the app continuously on a VDI or server so multiple users can access it:

#### 1. Install dependencies on the server

```bash
pip install streamlit pandas sentence-transformers groq python-dotenv openpyxl
```

#### 2. Run in headless mode (no browser auto-open)

```bash
nohup streamlit run app.py --server.port 8501 --server.headless true &
```

This keeps the app running after you disconnect from the session.

#### 3. Users access it via browser

```
http://<server-ip-address>:8501
```

#### 4. Recommended extras for reliability

| What | Why |
|---|---|
| Process manager (`systemd` on Linux, Windows Service) | Auto-restarts the app if it crashes |
| Reverse proxy (Nginx or Apache) | Adds HTTPS, clean URLs, load balancing |
| Firewall rule | Allow inbound traffic on port 8501 |

**Example `systemd` service file (Linux):**

```ini
[Unit]
Description=Employee Search Assistant
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/Search_Assistant
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port 8501 --server.headless true
Restart=always
EnvironmentFile=/path/to/Search_Assistant/.env

[Install]
WantedBy=multi-user.target
```

### Choosing an LLM for Scale

The current setup uses Groq's free tier (30 requests/min), which works for personal use and small teams. As your user base grows, you'll need to upgrade the LLM provider. The good news: **switching LLMs is a ~5 line change** in `app.py` since only the `ask_llm` function talks to the LLM.

| Provider | Model | Cost (per 1K queries) | Rate Limit | Best For |
|---|---|---|---|---|
| **Groq Free** (current) | Llama 3.3 70B | Free | 30 req/min | Personal use, testing |
| **Groq Paid** | Llama 3.3 70B | ~$0.05-0.10 | Much higher | Small teams (10-50 users) |
| **OpenAI** | GPT-4o-mini | ~$0.15 | 10,000 req/min | Hundreds to thousands of users |
| **Azure OpenAI** | GPT-4o-mini | ~$0.15 | Enterprise SLAs | Organizations using Azure |
| **AWS Bedrock** | Claude / Llama | ~$0.10-0.50 | Auto-scales | Organizations using AWS |
| **Self-hosted** | Llama 3 via vLLM | GPU server (~$1-3/hr) | Unlimited | Full control, data stays on-prem |

**Recommended path as you scale:**

```text
Solo / Testing          Small Team (10-50)        Org-wide (100s-1000s)        Enterprise
Groq Free        -->    Groq Paid             -->  OpenAI / Azure OpenAI   -->  Self-hosted or Cloud AI
(current)               (just upgrade key)         (change ~5 lines)           (full infra control)
```

### Key Considerations at Scale

**Data privacy:** If employee data is sensitive, consider self-hosting the LLM (Llama 3 via vLLM on a GPU server) so no data leaves your network. With API-based LLMs (Groq, OpenAI), employee records are sent to external servers as part of each query.

**Concurrent users:** Streamlit runs single-threaded by default. For many concurrent users, deploy with multiple Streamlit workers behind a load balancer, or migrate the backend to a framework like FastAPI.

**Cost control:** At 1,000 queries/day with OpenAI GPT-4o-mini, expect roughly $4-5/month. Costs scale linearly with usage. Self-hosting has a fixed server cost regardless of query volume.
