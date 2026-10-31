# ApprentiGuide

> Answers the questions of apprentices and students in France about their apprenticeship, work and residence rights, with the exact article or fact sheet it comes from, or says it does not know.

*Ask anything. Never apologise.*

## 1. User and problem
- **User:** an aivancity apprentice (or a first-year student who wants to become one, or an international student) with a question about leave, pay, the contract, working time, meal vouchers or their right to work.
- **Problem today:** the answer is spread over the Code du travail, the CESEDA and Service-Public.fr fact sheets. Students ask HR, the school or a general chatbot: HR answers a day later, a general chatbot answers without sources and cannot be checked, Service-Public.fr does not apply the rule to the person's case.
- **Value if it works:** an answer in seconds that the student can check (article and link), the right rule for their situation (age, year of contract, private employer), and, when the answer depends on the employer, exactly what to ask HR.

## 2. What the system does
- **Input:** a question in French or English, typed or spoken (local speech-to-text, the audio is not stored), with the conversation so far.
- **Output:** a short answer in the language of the question, every fact followed by its source id (`[L6222-35]`, `[F2918]`), and under it the sources with their link (Legifrance article or Service-Public.fr fact sheet), version date and the exact passage. When the texts do not answer: a fixed message saying where we looked and what to do next (rephrase, ask the school or HR).
- **Technique:** RAG with hybrid retrieval and abstention decided in code.
  - French rewrite of the question (gpt-oss-120b) for the full text search and the reranker.
  - Three retrieval signals fused by reciprocal rank fusion: article or fact sheet number written in the question, PostgreSQL French full text, dense search with **bge-m3** (multilingual, 1024 dimensions, pgvector).
  - Cross-encoder reranker **bge-reranker-v2-m3** on 30 candidates; 5 passages to the model, at most one per document; public-sector texts after the others unless the question is about the public sector.
  - Abstention in code: best reranker score under a threshold, or a named article that is not in the corpus.
  - Follow-up questions ("and in the second year?") are made complete from the conversation before retrieval.
  - Answer: **gpt-oss** on Groq (reasoning model, low reasoning effort), system prompt `prompts/system_v1.txt`; **Qwen3 8B** through Ollama as the local fallback (nothing leaves the laptop).
  - Why: most RAG failures are retrieval failures, and the reranker is the piece that matters most here (without it, Recall@1 falls from 0.90 to 0.43 on our draft set). Every choice and every rejected idea is measured in `docs/decisions_and_experiments.md`.

## 3. How to run it
Docker Desktop must be running.
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Groq key (console.groq.com/keys), POSTGRES_PASSWORD and DATABASE_URL, DB_PORT if 5432 is taken
docker compose --env-file .env -f db/docker-compose.yml up -d   # PostgreSQL with pgvector
python -m ingest.load           # chunks from data/raw (1434), no PISTE key needed
python -m rag.index             # vectors (bge-m3, local, about 5 min the first time; then only what changed)
```
Chat interface (Open WebUI, conversation history and voice input built in), in two terminals:
```bash
uvicorn api:app --app-dir app --host 0.0.0.0 --port 8001     # OpenAI-compatible API in front of the pipeline
docker run -d --name apprentiguide-webui -p 127.0.0.1:3080:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8001/v1 -e OPENAI_API_KEY=local \
  -e ENABLE_OLLAMA_API=False -e WEBUI_AUTH=False -e WEBUI_NAME=ApprentiGuide \
  -v apprentiguide-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
# open http://localhost:3080 and choose the model "apprentiguide"
```
`ANSWER_MODEL=ollama:qwen3:8b uvicorn ...` answers with the local model instead of Groq. A simpler interface without Docker: `streamlit run app/ui.py`.
Evaluation (writes a JSON report; see `eval/eval.py --help`):
```bash
python eval/eval.py --retrieval-only        # Recall@k, MRR, abstention: no answer model, rewrites cached
python eval/eval.py                         # full answers on eval/testset.jsonl
python eval/eval.py --baseline              # same model without documents
python eval/eval.py --model ollama:qwen3:8b # local answer model (ollama pull qwen3:8b)
```
`RAG_DEVICE=cpu` runs the embedding model and the reranker on the CPU (useful when a local LLM uses the Mac GPU).

## 4. Data
See `data/README.md`: Code du travail, CESEDA and Code de la sécurité sociale articles through the Legifrance API (PISTE), Service-Public.fr fact sheets (open data DILA, Licence Ouverte), how they were selected and cleaned. Fact sheets from the open data file of 25 Oct. 2026 (date shown under every answer); articles fetched on 25 and 30 Oct. 2026, each with its version date. No personal data in the corpus; user questions are not stored.

## 5. Evaluation (summary)
Test set: `eval/testset.jsonl`, 81 questions (70 by the evaluation lead, 11 added; 61 with an answer in the texts, 20 without), French and English, three families: answer in the texts, depends on the employer, no answer. Gold sources under review (`gold_added`, to be validated). 95 % Wilson intervals.

| Measure (31 Oct.) | Result |
|---|---|
| A gold source among the 5 passages given to the model (Recall@5) | 0.87 (0.76 to 0.93) |
| Gold source ranked first (Recall@1) | 0.57 (0.45 to 0.69) |

| System (37 questions drawn from the 81) | Gold source cited | NOT FOUND when no answer | Invented citations | Median latency |
|---|---|---|---|---|
| Baseline (prompt only) | to be run on 31 Oct. | | | |
| gpt-oss-120b, prompt v1 (30 Oct.) | 0.70 (0.52 to 0.84) | 8/9 | 0/36 | 13 s |
| Qwen3 8B local, prompt v2, with reasoning | 0.68 (0.49 to 0.82) | 7/9 | 0/37 | 49 s |
| Qwen3 8B local, prompt v2, without reasoning | 0.64 (0.46 to 0.79) | 7/9 | 0/37 | 31 s |
| llama3.2 3B local, prompt v2 | 0.64 (0.46 to 0.79) | 7/9 | 2/37 | 22 s |

Full report: `docs/evaluation_report.md`. System card: `docs/system_card.md`. Every decision and experiment, including the ideas we measured and removed (labels per document, agent-like steps in a fixed pipeline): `docs/decisions_and_experiments.md`.

## 6. AI tools used to build this project
| Tool | Used for |
|---|---|
| Claude Code (Anthropic), by Johanna Fokui | Retrieval and evaluation code (`rag/`, `app/pipeline.py`, `eval/eval.py`), 11 questions added to the test set (ids A01 to A11, tagged `added_by: assistant`), review of gold sources (`gold_added`, each with a quote from the text, checked against the database), drafts of the documentation. Every result was checked by running it; the decisions are ours. |
| *(each member adds the tools they used)* | |

## 7. Team
| Member | Main role | Also worked on |
|---|---|---|
| Julie Abouem | data (Legifrance API, Service-Public.fr open data, PostgreSQL) | corpus extensions requested from the test set |
| Johanna Fokui | retrieval / model | evaluation tooling, test set additions |
| Nafissatou Ndiaye | evaluation and safety | test set (70 questions), red-team set (28 cases) |
| Nafissatou Ndiaye | product and demo | |
