# Retrieval: decisions and experiments

Log of the retrieval and answer pipeline (`rag/`, `app/pipeline.py`, `eval/eval.py`): what we decided, what we tried, what we observed. Newest entries at the end of each section. Every number here comes from a run on the database built by `ingest.load` (1372 chunks, 850 documents, data files of 2026-10-25).

**Status, 30 Oct. 2026:** pipeline built and unit tested (48 tests). Only smoke tests so far (5 and 3 questions): they show the pipeline works, they do not measure quality. The measured results come with the test set.

## 1. Decisions

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| D1 | Four retrieval signals: exact article or fiche number, French full text, dense (bge-m3) on chunks, dense on short labels per document. Fused by reciprocal rank fusion (k = 60). | Each signal fails differently: full text wins on legal terms and article numbers, dense on paraphrases and English questions, labels bridge everyday words and legal wording. Ranks are comparable, scores are not. | Dense only: best Recall@1 in the 7 Oct. lab, but no exact match on article numbers. |
| D2 | Cross-encoder reranker (bge-reranker-v2-m3) on the 30 fused candidates, top 5 to the model. | 7 Oct. lab: hybrid alone had Recall@1 0.79 (worse than dense, 0.93); hybrid + reranker reached 1.00. RRF gives the same weight to a bad list; the reranker repairs the order. | No reranker: 5x faster, measured in the ablation (section 3). |
| D3 | Abstention (NOT FOUND) decided in code: if the best reranker score is under a threshold, the model is not called. | 7 Oct. lab: the "reply NOT FOUND" instruction gives the format, not the judgement; the model described the role instead of abstaining. A rule that must hold is code, not a prompt. | Prompt-only abstention: kept as a second layer (rule 6 of the prompt). |
| D4 | One chunk per article (Legifrance) and per section of fiche (service-public), the article or fiche number as citation id. | Done by the data pipeline; the number is what the answer cites and what the user can check. | Fixed-size chunks: would cut a rule from its conditions. |
| D5 | Labels per document, extracted by an LLM then cleaned by code: 1 to 3 categories (English, the topic), up to 12 keywords (original French wording), up to 8 concepts (English, abstraction of a keyword). Stored in a separate index (`doc_labels`). | A question in English or in everyday words ("my boss", "lunch vouchers") shares few words with the legal text. Short English handles are a bridge; keeping keywords and content in separate indexes stops short labels from polluting content search. | Labels in the same index as the chunks. |
| D6 | The question is rewritten in French (one LLM call) for the reranker and the full text search; the dense signals use the original question. | Observed (smoke test): the reranker gave the right article 0.035 for an English question and 0.004 for an off-topic one; any fixed threshold would drop correct English answers. | No rewrite, lower threshold: cannot separate English questions from off-topic ones. |
| D7 | Article or fiche numbers that the user did not write are removed from the rewrite, in code. | Observed: the rewrite added "(article L. 6323-1)", which was not in the question and would send full text search to the wrong article. | Prompt rule only: kept too, but not trusted alone. |
| D8 | If the user names an article that is not in the corpus, the model is told so and must say it, not answer from a neighbouring article. | A plausible answer from the wrong article is a silent error: the user cannot see it. | |
| D9 | System prompt v1: every fact cited with its id; figures and conditions, never "see the document"; the rule matching the user's situation first; general rule plus what to ask HR or the school when the answer depends on the employer; partial answers say what is missing. | Turns retrieval into an answer a person can act on and check. The company-specific case comes from our pitch story (meal vouchers). | |
| D10 | `max_tokens` 2000 and `reasoning_effort: low` for gpt-oss-20b. | Reasoning model: its reasoning tokens count in `max_tokens`; 400 (starter value) can cut the answer. | |
| D11 | Labels made by a local model (llama3.2 on Ollama), answers and evaluation on Groq (gpt-oss-20b). | Groq free plan for gpt-oss-20b: 8000 tokens per minute and 1000 requests per day. Labels need about 850 calls of 1500 to 2500 tokens: most of a day's quota, nothing left for evaluation. See experiment E4. | Labels on Groq: 80 rate-limit errors in one run. |
| D12 | Indexes are incremental: each vector and label set stores a hash of its text; a reload recomputes only what changed and drops what left the corpus. No scheduled refresh. | The corpus is a dated snapshot (files of 2026-10-25), which also keeps evaluation runs comparable. The risk, "faithful is not correct" (a faithful citation of an outdated text), goes in the system card with the snapshot date. | Daily refresh: useful for a live product, not for a graded evaluation. |
| D13 | PostgreSQL with pgvector (image `pgvector/pgvector`), host port set by `DB_PORT` (default 5432). | One database for text, full text index, vectors and labels; the port avoids a clash with another local PostgreSQL. | A separate vector store (Chroma). |

Not in the 3 Nov. scope: an agent that chooses its own search tools (planned for the Agentic AI course), boosting by user profile (international, age, programme), the school regulations (written permission pending).

## 2. Label extraction: iterations

Checked by reading random samples of labels next to the text of the article.

| Version | Change | What we saw | Next step |
|---|---|---|---|
| v1 | First prompt, gpt-oss-20b on Groq | Legal plumbing as labels: "Effective Date", "1er mai 2021", "décret n° 2020-1734", "article R. 422-9". They match every article that has an entry-into-force note. | Deterministic filters (references, decree numbers, dates, entry into force) + prompt rule. Unit test on these exact cases. |
| v2 | Filters, low reasoning effort | Keywords concrete and faithful ("un mois avant son départ", "indemnité de fin de mission"). But "Paid Leave" on 66 of 120 documents, including caregiver and sabbatical leave, and "Notification" used as a catch-all. Cause: known categories shown by frequency, so the most used one attracts the next documents. | Known categories in alphabetical order, generic categories dropped in code, "topic, not procedure" rule. |
| v3 | Topic categories | Example categories in the prompt copied as is ("Apprenticeship Contract" on an article about building-trade leave funds). Short articles ("Le régime prévu par la présente section...") give the model no topic. | Remove example categories; give the model the last 4 section titles of the document ("Section 5 : Caisses de congés payés > ... bâtiment et travaux publics"). |
| v4 | Section titles in the prompt, local model | See E4 for the model choice. | Full run, then ablation with and without labels. |

## 3. Experiments

### E1. Retrieval smoke test (5 questions, signals exact + full text + dense, no labels, no rewrite)

| Question | Best reranker score | Top source | Comment |
|---|---|---|---|
| Combien de jours de congés ai-je en apprentissage ? | 0.996 | F2918 (congés de l'apprenti) | L6222-35 (5 days for exams) 3rd |
| My employer cut my meal vouchers this month, is that allowed? | 0.035 | R3262-7 | right topic, very low score (English vs French text) |
| Que dit l'article L6222-35 ? | 0.912 | L6222-35 | exact signal |
| I am a first-year international student. Can I do an apprenticeship this year? | 0.027 | R5221-7 | right article, very low score |
| Quel est le menu de la cantine de l'école ? (no answer) | 0.004 | F38630 | noise only, as expected |

Latency: about 4 s per question after loading the models (first call 21 s). Led to D6.

### E2. Evaluation smoke test (3 questions, retrieval only, with rewrite)

| Id | Best score | Rank of the gold source | French rewrite |
|---|---|---|---|
| s1 (exam leave, French) | 0.94 | 1 | "... (article L. 6323-1)": invented number, led to D7 |
| s2 (meal vouchers, English) | 0.012 | 1 | "Mes tickets restaurant ont été supprimés ce mois-ci, est-ce légal?" |
| s3 (canteen menu, no answer) | 0.004 | none | rewrite failed (Groq quota used by labels), fell back to the original question |

Even in French, s2 scores 0.012: the threshold must be set on the test set, not guessed. Three questions give intervals from 34 to 100 %: nothing can be concluded from them.

### E3. Groq quota (free plan, gpt-oss-20b)

Response headers: 1000 requests per day, 8000 tokens per minute. At about 2000 tokens per labelled document the run is capped at 4 to 5 documents per minute; above that, HTTP 429 (80 documents skipped in one run).

### E4. Model for labels (same 4 documents)

| Document | mistral 7B (local) | llama3.2 3B (local) |
|---|---|---|
| D3141-14 (leave funds, building trade) | category = the copied section path | "Affiliation Rules"; weak keywords |
| D3142-14 (sabbatical leave deadlines) | category in French ("Congé sabbatique") | "Sabbatical Leave", "Work-Life Balance" |
| L6222-35 (exam leave) | good | good: "Pre-Exam Leave", "cinq jours supplémentaires" |
| F21059 (meal vouchers) | category = copied path | "Meal Vouchers", "Titres-restaurant", "Chèque déjeuner" |
| Time per document | 7 to 23 s | 2 to 8 s |

llama3.2 chosen: cleaner output and 3 times faster (about 1 h for 850 documents). A category containing ">" or ":" (a copied path) is now rejected in code.

## 4. Next measurements

1. Test set (60+ questions: core, paraphrase, English, unanswerable, company-specific), validated by the evaluation lead.
2. `python eval/eval.py --retrieval-only`: Recall@1/3/5, MRR, abstention threshold.
3. Ablation, one change at a time: without labels, without full text, without reranker, without rewrite.
4. Baseline (`--baseline`) against the full system on the same questions, with 95 % intervals.
