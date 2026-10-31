# ApprentiGuide

> Answers the questions of apprentices and students in France about their apprenticeship, work and residence rights, with the exact article or fact sheet it comes from, or says it does not know.

*Ask anything. Never apologise.*

## 1. User and problem
- **User:** an aivancity apprentice (or a first-year student who wants to become one, or an international student) with a question about leave, pay, the contract, working time, meal vouchers or their right to work.
- **Problem today:** the answer is spread over the Code du travail, the CESEDA and Service-Public.fr fact sheets. Students ask HR, the school or a general chatbot: HR answers a day later, a general chatbot answers without sources and cannot be checked, Service-Public.fr does not apply the rule to the person's case.
- **Value if it works:** an answer in seconds that the student can check (article and link), the right rule for their situation (age, year of contract, private employer), and, when the answer depends on the employer, exactly what to ask HR.

## 2. What the system does
- **Input:** a question in French or English, typed or spoken (local speech-to-text, the audio is not stored), with the conversation so far.
- **Output:** a short answer in the language of the question, every fact followed by a numbered citation ([1], [2]); each number opens a source card: the article or fact sheet (Legifrance or Service-Public.fr), its version date and link, the sentences that answer the question, then the full text. The answer is written as it comes, with status lines while searching and the model's reasoning in a folded block. When the texts do not answer: a fixed message saying where we looked and what to do next (rephrase, ask the school or HR).
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
Chat interface (Open WebUI 0.12.5, conversation history and voice input built in; settings in the command, so it starts the same everywhere), in two terminals:
```bash
uvicorn api:app --app-dir app --host 0.0.0.0 --port 8001     # OpenAI-compatible API in front of the pipeline
docker run -d --name apprentiguide-webui -p 127.0.0.1:3080:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8001/v1 -e OPENAI_API_KEY=local \
  -e ENABLE_OLLAMA_API=False -e WEBUI_AUTH=False -e WEBUI_NAME=ApprentiGuide \
  -e ENABLE_PERSISTENT_CONFIG=False -e ENABLE_TITLE_GENERATION=True -e ENABLE_TAGS_GENERATION=False \
  -e ENABLE_FOLLOW_UP_GENERATION=False -e ENABLE_AUTOCOMPLETE_GENERATION=False \
  -e DEFAULT_PROMPT_SUGGESTIONS="$(cat app/webui_suggestions.json)" \
  -v apprentiguide-webui:/app/backend/data ghcr.io/open-webui/open-webui:v0.12.5
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
The evaluation is led by Nafissatou Ndiaye: test set (`eval/testset.jsonl`, 70 questions in French and English: answer in the texts, depends on the employer, no answer), red-team set (`eval/redteam.jsonl`, 28 cases), final runs, report and system card. Results to come in `docs/evaluation_report.md`, `docs/red_team_log.md` and `docs/system_card.md`.

| System | Main metric | NOT FOUND rate | Median latency |
|---|---|---|---|
| Baseline (prompt only) | | | |
| Our system | | | |

The development measurements made while building the retrieval (to choose its settings) are in `docs/decisions_and_experiments.md`, with every decision and the ideas measured and removed. They are not the project's evaluation.

## 6. AI tools used to build this project
| Tool | Used for |
|---|---|
| Claude Code (Anthropic), by Johanna Fokui | Retrieval, interface and evaluation tooling code (`rag/`, `app/`, `eval/eval.py`, `eval/voice_eval.py`), development measurements, proposals for the test set (11 questions with ids A01 to A11 and `gold_added` sources, each with a quote from the text; to be accepted or removed by the evaluation lead), drafts of the documentation. Every result was checked by running it; the decisions are ours. |
| *(each member adds the tools they used)* | |

## 7. Team
| Member | Main role | Also worked on |
|---|---|---|
| Julie Abouem | data (Legifrance API, Service-Public.fr open data, PostgreSQL) | corpus extensions requested from the test set |
| Johanna Fokui | retrieval / model | chat interface and voice input, tooling to run the evaluation |
| Nafissatou Ndiaye | evaluation and safety | test set (70 questions), red-team set (28 cases) |
| Nafissatou Ndiaye | product and demo | |
