# Retrieval: decisions and experiments

Log of the retrieval and answer pipeline (`rag/`, `app/pipeline.py`, `eval/eval.py`): what we decided, what we tried, what we kept, what we removed and why. Every number comes from a run on the database built by `ingest.load` (1372 chunks, 850 documents, data files of 2026-10-25). Raw results of each run are in `eval/runs/`.

**Status, 30 Oct. 2026, 17:00.** On the full test set (81 questions, gold sources reviewed): Recall@1 0.55, Recall@5 0.80 (E10). Earlier, at 16:30: Pipeline built and unit tested (44 tests). On the draft test set (36 questions), retrieval reaches Recall@1 0.90 (95 % CI 0.74 to 0.97), Recall@3 0.97, MRR 0.92 (E7). The ablation (E8) showed that labels per document bring nothing here: they were removed (section 3). The evaluation lead's test set (70 questions, `eval/testset.jsonl`) is now the reference; its first run (E9) shows that gold sources must list every text that answers, not only the Code article, before its numbers mean anything.

## 1. The pipeline as it is

```
question
  -> French rewrite (gpt-oss-120b, translation only; numbers the user did not write removed in code)
  -> 3 signals, 20 candidates each:
       exact     article or fiche number written in the question (L6222-35, F2918)
       fulltext  PostgreSQL French full text, any word may match, on the French rewrite
       dense     bge-m3 on the chunks, on the original question
  -> reciprocal rank fusion (k = 60), 30 candidates
  -> reranker bge-reranker-v2-m3 on the French rewrite, top 5
  -> abstention in code: best score < 0.015, or a named article missing from the corpus -> NOT FOUND
  -> answer by gpt-oss-20b with system prompt v1 (every fact cited)
```

## 2. Decisions

| # | Decision | Why | Alternative not taken |
|---|---|---|---|
| D1 | Three retrieval signals (exact number, French full text, dense bge-m3), fused by reciprocal rank fusion (k = 60). | Each fails differently: full text wins on legal terms and article numbers, dense on paraphrases and English questions, exact on a cited article. Ranks are comparable, scores are not. Without full text, Recall@1 drops from 0.90 to 0.87 (E8). | Dense only: best Recall@1 in the 7 Oct. lab, but no exact match on article numbers. |
| D2 | Cross-encoder reranker (bge-reranker-v2-m3) on the 30 fused candidates, top 5 to the model. | The single most useful piece: without it, Recall@1 falls from 0.90 to 0.43 and abstention from 6/6 to 1/6 (E8). Same lesson as the 7 Oct. lab (hybrid 0.79, hybrid + reranker 1.00). | No reranker: faster, far worse. |
| D3 | Abstention decided in code, not by the model. | 7 Oct. lab: the "reply NOT FOUND" instruction gives the format, not the judgement. A rule that must hold is code. | Prompt-only abstention: kept as a second layer (rule 6 of the prompt). |
| D4 | One chunk per article and per section of fiche, the article or fiche number as citation id. | Done by the data pipeline; the number is what the answer cites and what the user can check. | Fixed-size chunks: would cut a rule from its conditions. |
| D5 | The question is rewritten in French for the reranker and the full text search; the dense signal uses the original question. | The reranker scored the right article 0.035 for an English question and 0.004 for an off-topic one (E1): no threshold could separate them. | No rewrite. |
| D6 | The rewrite has its own model setting: gpt-oss-120b on Groq, prompt "translate, keep everything, do not answer"; quotes and refusals rejected in code, numbers the user did not write removed in code. | About 200 tokens per question, its own daily quota. The rewrite once added "(article L. 6323-1)", which was not in the question (E2), and once returned "I'm sorry, but I can't help with that." | llama3.2 locally: answered instead of translating (E6). |
| D7 | Abstention rules: reranker threshold 0.015, and NOT FOUND (with the reason) when the question names an article that is not in the corpus. | On the draft test set, unanswerable questions scored at most 0.009, answerable ones at least 0.024; the invented article L9999-1 scored 0.386 on its neighbours, so a threshold alone cannot catch it (E7). The threshold was set on the test set itself: optimistic until checked on the evaluation lead's set. | One threshold only. |
| D8 | System prompt v1: every fact cited with its id; figures and conditions, never "see the document"; the rule matching the user's situation first; general rule plus what to ask HR or the school when the answer depends on the employer; partial answers say what is missing. | Turns retrieval into an answer a person can act on and check. The company-specific case comes from our pitch story (meal vouchers). | |
| D9 | `max_tokens` 2000 and `reasoning_effort: low` for gpt-oss. | Reasoning model: its reasoning tokens count in `max_tokens`; 400 (starter value) can cut the answer. | |
| D10 | The vector index is incremental: each vector stores a hash of its text; a reload recomputes only what changed and drops what left the corpus. No scheduled refresh. | The corpus is a dated snapshot (files of 2026-10-25), which keeps evaluation runs comparable. The risk, "faithful is not correct" (a faithful citation of an outdated text), goes in the system card with the snapshot date. | Daily refresh: useful for a live product, not for a graded evaluation. |
| D11 | PostgreSQL with pgvector (image `pgvector/pgvector`), host port set by `DB_PORT` (default 5432). | One database for text, full text index and vectors; the port avoids a clash with another local PostgreSQL. | A separate vector store (Chroma). |
| D12 | `RAG_DEVICE=cpu` to run the embedding model and the reranker on the CPU. | On a Mac, a local LLM using the GPU made them wait with 0 % CPU (E5). | |
| D13 | Labels per document: tried, measured, removed (section 3). | No gain in the ablation, one hour of local LLM per full index. | |
| D14 | The answer language is decided in code (French if the French rewrite left the question unchanged, else English) and stated at the end of the request to the model. | E11: 9 of 9 English questions answered in French with French sources, despite the prompt rule. | Prompt rule only. |
| D15 | Prompt v2 (`prompts/system_v2.txt`, v1 kept as measured): answer the exact question (a maximum is a maximum); for company-dependent questions give what the law sets or say it sets nothing, then what to ask HR or the school, never NOT FOUND if a source covers the topic; NOT FOUND only if no source is about the topic; ids in square brackets. | E11 reading. To be measured against v1 on the same questions before switching `prompt_file`. | |
| D16 | NOT FOUND is shown as a fixed message written in code, in the language of the question: where we looked (the codes and Service-Public.fr) and what to do (rephrase, ask the school or HR), plus the missing article when one was named. NOT FOUND stays an internal flag (`not_found`, with its reason) for the evaluation. | A bare "NOT FOUND" tells the user nothing; a fixed text cannot be hallucinated. | Message written by the model. |
| D18 | At most 1 passage per article or fiche among the 5 given to the model (`max_per_doc`). | E12: three sections of one fiche often took three of the five places; with the cap the model sees 5 distinct documents and Recall@5 goes from 0.80 to 0.85. To check when reading answers: a long fiche now contributes one section. | 2 per document (0.83). |
| D19 | Public-sector texts (section title "secteur public", fiche F3059) go after the others unless the question is about the public sector (`sector_priority`). | E14: they took 36 of 400 places while one question of 81 is about the public sector; with the rule 6 questions improve, none gets worse. Same idea as a country priority in an HR assistant: the user's own rules first. Detected by section title, not by article number (R6227 is a chapter of penalties, not the public sector). | Removing them: kept, in case nothing else answers. |
| D20 | Not kept: wider candidate pools (E12) and following references between articles (E13). | No measurable gain. | |
| D17 | Sources under each answer come from the database, never from the model: link (Legifrance article or Service-Public fiche), label, version date, passages; ids cited but not retrieved are flagged as unverified and counted. The page shows the Licence Ouverte attribution with the date of the data files. | Users must be able to check the text themselves; Licence Ouverte requires the source and its date. | Links written by the model. |

Not in the 3 Nov. scope: an agent that chooses its own search tools (planned for the Agentic AI course), boosting by user profile (international, age, programme), the school regulations (written permission pending).

## 3. Tried and removed: labels per document

### Why we tried it

Johanna works on a production assistant over internal HR policies (several thousand pages, several languages). There, a pattern made a large difference: for every page, an LLM writes short labels (1 to 3 abstract categories in English, about 10 concrete keywords in the original language, 5 to 8 English concepts that abstract the keywords), cleaned by deterministic rules and stored in their own vector index next to the content. A question in another language or in everyday words ("lunch vouchers", "my boss") then meets an English concept ("Meal Allowance") that points to the right page, without any translation step. Our users ask in English and in everyday French about texts written in legal French, so the same bridge looked useful.

### What we built

`rag/labels.py` (prompt, JSON parsing, deterministic filters), a `doc_labels` table with its own HNSW index, and a fourth retrieval signal: documents ranked by their closest label, each represented by its chunk closest to the question. Labels were checked by reading random samples next to the text of the article:

| Version | Change | What we saw |
|---|---|---|
| v1 | gpt-oss-20b on Groq | Legal plumbing as labels: "Effective Date", "1er mai 2021", "décret n° 2020-1734", "article R. 422-9". They match every article with an entry-into-force note. Fixed with deterministic filters. |
| v2 | Filters, low reasoning effort | Keywords concrete and faithful ("un mois avant son départ"). But "Paid Leave" on 66 of 120 documents, including caregiver and sabbatical leave: known categories were shown by frequency, so the most used one attracted the next documents. Fixed: alphabetical order, generic categories dropped. |
| v3 | Topic categories | Example categories in the prompt copied as is; short articles ("Le régime prévu par la présente section...") gave the model no topic. Fixed: no examples, the last 4 section titles of the document in the prompt. |
| v4 | Local model (E3, E4) | Groq's free quota could not take 850 documents (E3), so llama3.2 on Ollama (better than mistral 7B on the same documents, E4). |
| v5 | llama3.2 on 850 documents | Keywords mostly good ("titre-mobilité", "congé sabbatique"); categories weak: "Ordre public" (a heading of the code) on 84 documents, "Notification of Decision" on 69, some wrong ("Work Permit" on a paid-leave article). Code headings and vague words filtered (227 of 4275 labels). |

### What the ablation showed (E8)

With labels (823 of 850 documents labelled): Recall@1 0.90, Recall@3 0.93, MRR 0.92. Without: Recall@1 0.90, Recall@3 0.97, MRR 0.92. No gain, slightly worse at rank 3.

### Why it brings nothing here

- **The embedding model already bridges languages.** The production assistant used an English-centred embedding model; bge-m3 is multilingual, so "meal vouchers" is already close to "titres-restaurant" (E1: the right article was ranked first for English questions before any label existed).
- **The topic is already in the chunk.** Each embedded chunk starts with its place in the code ("Titre IV : Congés payés et autres congés > ... > Congé sabbatique"). Those human-written section titles are better topic labels than any category the LLM produced (v3, v5).
- **The reranker reads the full text.** Whatever the labels add to the candidates, the cross-encoder decides the order on the article itself.
- **Cost:** about one hour of local LLM for a full index, and the first attempt used up a day of the Groq quota (E5).

Labels could be worth trying again with a weaker or English-only embedding model, a much larger corpus with poorer titles, or as a discovery tool for the agent ("which topics exist?"). The code is in the history at commit `6158cbc` (`rag/labels.py`, `index_labels` in `rag/index.py`, `labels_signal` in `rag/retrieve.py`).

## 4. Experiments

### E1. Retrieval smoke test (5 questions, exact + full text + dense, no rewrite)

| Question | Best reranker score | Top source | Comment |
|---|---|---|---|
| Combien de jours de congés ai-je en apprentissage ? | 0.996 | F2918 (congés de l'apprenti) | L6222-35 (5 days for exams) 3rd |
| My employer cut my meal vouchers this month, is that allowed? | 0.035 | R3262-7 | right topic, very low score (English vs French text) |
| Que dit l'article L6222-35 ? | 0.912 | L6222-35 | exact signal |
| I am a first-year international student. Can I do an apprenticeship this year? | 0.027 | R5221-7 | right article, very low score |
| Quel est le menu de la cantine de l'école ? (no answer) | 0.004 | F38630 | noise only, as expected |

About 4 s per question after loading the models (first call 21 s). Led to D5.

### E2. Evaluation smoke test (3 questions, retrieval only, with rewrite)

| Id | Best score | Rank of the gold source | French rewrite |
|---|---|---|---|
| s1 (exam leave, French) | 0.94 | 1 | "... (article L. 6323-1)": invented number, led to D6 |
| s2 (meal vouchers, English) | 0.012 | 1 | "Mes tickets restaurant ont été supprimés ce mois-ci, est-ce légal?" |
| s3 (canteen menu, no answer) | 0.004 | none | rewrite failed (quota), fell back to the original question |

### E3. Groq free plan limits (gpt-oss-20b and gpt-oss-120b)

Per model: 1000 requests per day, 8000 tokens per minute (response headers), and 200 000 tokens per day, which the headers do not show (error message: "tokens per day (TPD): Limit 200000, Used 199869"). At about 2000 tokens per labelled document the label run was capped at 4 to 5 documents per minute, with 80 documents rejected (HTTP 429) in one run, and it used up the day's tokens of gpt-oss-20b. Budget: about 3000 tokens per answered question, so one full answer run of 36 questions (about 110 000 tokens) per model per day and per key; retrieval-only runs use about 200 tokens per question.

### E4. Local model for labels (same 4 documents)

| Document | mistral 7B | llama3.2 3B |
|---|---|---|
| D3141-14 (leave funds, building trade) | category = the copied section path | "Affiliation Rules"; weak keywords |
| D3142-14 (sabbatical leave deadlines) | category in French ("Congé sabbatique") | "Sabbatical Leave", "Work-Life Balance" |
| L6222-35 (exam leave) | good | good: "Pre-Exam Leave", "cinq jours supplémentaires" |
| F21059 (meal vouchers) | category = copied path | "Meal Vouchers", "Titres-restaurant", "Chèque déjeuner" |
| Time per document | 7 to 23 s | 2 to 8 s |

### E5. GPU contention

During the local label run, an evaluation run sat at 0 % CPU for 10 minutes: the embedding model and the reranker (on the Mac GPU) waited behind the LLM. With `RAG_DEVICE=cpu`, about 11 s per question. Led to D12.

### E6. Model for the rewrite (same questions)

| Question | llama3.2 (local) | gpt-oss-120b (Groq) |
|---|---|---|
| q001 exam leave | "Vous avez 5 jours de congé..." (answers, and puts the answer in the query) | the question, unchanged |
| q011 diploma, English | "Puis-je faire fin à mon stage avant..." (wrong meaning) | "Mon apprentissage peut-il se terminer plus tôt si j'ai déjà obtenu mon diplôme ?" |
| q021 meal voucher share | "L'employeur doit payer la moitié..." (answers) | "Quelle part d'un ticket restaurant l'employeur doit-il payer ?" |
| Time | 1 to 57 s (GPU busy) | 0.2 to 1.3 s |

### E7. Retrieval on the draft test set (`eval/testset_draft_assistant.jsonl`, 36 questions: 30 answerable, 6 without answer)

| Run | Rewrite | Recall@1 | Recall@3 | MRR |
|---|---|---|---|---|
| `retrieval_no-labels.json` | gpt-oss-20b, 28 of 36 empty (quota) | 0.767 | 0.933 | 0.833 |
| `retrieval_no-labels_v2.json` | gpt-oss-20b, 23 of 36 empty (quota) | 0.800 | 0.933 | 0.856 |
| `retrieval_no-labels_v3.json` | gpt-oss-120b, none empty | 0.800 | 0.967 | 0.867 |
| same run, gold sources corrected | | **0.900** (CI 0.744 to 0.965) | 0.967 (CI 0.833 to 0.994) | 0.922 |

Gold sources corrected after reading the retrieved texts: F37900 for q022 and q023, F20740 for q026, F2216 for q013 (each answers the question better than the source first listed). Remaining misses at rank 1: q002 ("j'ai un partiel, mon patron doit-il me laisser réviser ?": exam leave never found), q005 (caregiver leave article first, F2258 third), q027 (F21059 third). Unanswerable questions scored 0.000 to 0.009, except the invented article L9999-1 at 0.386 (led to D7).

### E8. Ablation, one change at a time (draft test set, gold sources corrected, `ablation_*.json`)

| Variant | Recall@1 (95 % CI) | Recall@3 | MRR | Unanswerable abstained |
|---|---|---|---|---|
| exact + full text + dense + labels | 0.900 (0.744 to 0.965) | 0.933 | 0.919 | 6/6 |
| **exact + full text + dense (kept)** | 0.900 (0.744 to 0.965) | 0.967 | 0.922 | 6/6 |
| exact + dense + labels (no full text) | 0.867 (0.703 to 0.947) | 0.933 | 0.889 | 6/6 |
| all signals, no reranker | 0.433 (0.274 to 0.608) | 0.900 | 0.639 | 1/6 |

Thirty answerable questions give wide intervals: the differences between the first three rows are within noise; the reranker's is not.

### E9. First run on the evaluation lead's test set (`eval/testset.jsonl`, 70 questions, `nafi_retrieval.json`)

8 questions have gold sources that are all outside the corpus (durée du travail L3121-18 and L3121-20, repos L3132-1 and L3132-3, travail de nuit L3163-1, médecine du travail L4624-1, L2254-1, and two fiches "à identifier"): reported apart as corpus gaps. On the other 62: Recall@1 0.12, Recall@3 0.35, Recall@5 0.49.

These numbers mostly measure how narrow the gold sources are, not retrieval: in 23 of the 38 misses a service-public fiche that states the rule (often F2918, "Contrat d'apprentissage") is ranked first while only the Code article is listed as gold; Q02 lists L6222-7 while L6222-7-1, the article on the length of the contract, is retrieved. Real abstention problem: "C'est quoi le numéro de mon contrat d'apprentissage ?" scores 0.909 on F2918, which speaks about contracts but not about the user's own contract number. That one must be caught at answer time (rule 6 of the prompt), not by the threshold.

### E10. Retrieval on the full test set (81 questions: the evaluation lead's 70 plus 11 from the draft; `testset81_retrieval.json`)

Gold sources reviewed first: for each answerable question, the 5 retrieved passages were read and a passage was added as gold (`gold_added`, with a quote as evidence) only if it states the rule of the reference answer; replacement articles named in the review were checked in the database (all present except L3121-20, a corpus gap). The evaluation lead's fields are unchanged; `--lead-gold-only` ignores the additions.

| Gold sources | Recall@1 (95 % CI) | Recall@3 | Recall@5 (95 % CI) | MRR |
|---|---|---|---|---|
| evaluation lead's only | 0.23 | 0.42 | | |
| with reviewed additions | **0.55** (0.43 to 0.67) | 0.75 | **0.80** (0.68 to 0.88) | 0.65 |

Abstention with the threshold of D7: 14 of 20 unanswerable questions abstained, 3 of 60 answerable ones wrongly abstained. Per family (Recall@1, or abstention for "no answer"): in corpus 0.54, company dependent 0.57, no answer 0.70; English 0.65, French 0.57. One question left as corpus gap (Q62, work on Saturday: L3132 not in the corpus).

This set is much harder than the draft (0.90): more paraphrases, conditions and company-dependent cases. The number that matters for the answer is Recall@5: in 80 % of answerable questions the model sees at least one source that states the rule. The review marked cut-off passages as uncertain rather than correct (for example the age section of F2918 for Q01), so Recall@1 is a lower bound.

### E11. First answer run (37 questions drawn from the 81, all families and both languages; gpt-oss-120b, prompt v1; `answers37_120b_prompt-v1.json`)

| Measure | Result (95 % CI) |
|---|---|
| A gold source is among the 5 given to the model | 0.78 (0.59 to 0.89) |
| The answer cites a gold source | 0.70 (0.52 to 0.84) |
| Answers when an answer exists | 0.93 (0.77 to 0.98) |
| NOT FOUND when no answer exists | 8/9 (0.57 to 0.98) |
| Answers citing an id that was not given (invented) | 0/36 |
| Median latency | 13 s |

Read by hand, question by question:
- **Citations.** The model writes ids as `[ F2918 ]`, `【F33754】`, `[F2918 ; D6222-1]` or wrapped in invisible characters; the first parser missed them (cited gold 0.52 before the fix, 0.70 after). Fixed in code (`cited_ids`), unit tested on these exact forms.
- **Language.** The 9 English questions answered were all answered in French, although the prompt asked for the language of the question. Now decided in code (D14).
- **Company-dependent questions.** Q58 (bonus, 13th month) and Q16 (company health plan) got NOT FOUND instead of "the law sets no general rule / here is the rule, ask HR for ...". A10 (our pitch story, fewer meal vouchers this month) gave secondary reasons (25 euros a day, order of use) but not the key rule (one voucher per meal within the working day, so none on days of absence), and never said to ask HR. Prompt v2 targets this (D15).
- **Exact question.** Q36 (maximum hours per week) answered the legal 35 hours instead of the maximum, with F1911 among the sources. Prompt v2 rule 2.
- Q58 still abstains with v2: no retrieved source is about bonuses (the relevant F2918 section, "L'apprenti est-il un salarié comme les autres ?", is not retrieved): a retrieval miss, not a prompt one.

### E12. Wider search and passages per document (81 questions, retrieval only, cached rewrites: no tokens)

| Variant | R@1 | R@3 | R@5 | R@8 | R@10 |
|---|---|---|---|---|---|
| 20 candidates per signal, 30 reranked (current) | 0.55 | 0.77 | 0.85 | 0.88 | 0.88 |
| 40 and 60 | 0.55 | 0.77 | 0.87 | 0.90 | 0.90 |
| 60 and 100 | 0.55 | 0.77 | 0.87 | 0.90 | 0.90 |

(Recall@k here counts distinct documents among 10 passages.) Wider pools: one question gained, within noise, slower: not kept. But the model only received 5 passages, often several sections of one fiche (median 4 distinct documents). Capping passages per document:

| Variant (5 passages to the model) | R@5 | Distinct documents (median) |
|---|---|---|
| no cap | 0.80 | 4 |
| at most 2 per document | 0.83 | 4 |
| **at most 1 per document (kept)** | **0.85** | 5 |

### E13. Following references between articles (agent-like step, in code)

459 legislative chunks cite another article ("prévu à l'article L. 6222-29"). Adding up to 2 cited articles from the top 3 passages: no question gained a gold source (Recall@5 0.85 with and without). Not kept. The answers are not lost in the references of the top passages.

### E14. Sector priority

Remaining misses read one by one: public-sector texts (F3059, L6227) appeared in 6 of the 9 questions without any gold source, because the word "apprenti" pulls every apprenticeship text. With public-sector texts ranked after the others (unless the question mentions the public sector): R@1 0.55 to 0.57, R@5 0.85 to 0.87, MRR 0.66 to 0.68; better on Q21, Q25, Q28, Q31, Q32, Q37; worse on none; Q64 (the one public-sector question) still at rank 1.

Remaining misses after E14 (in the 5 sources given): Q03, Q10, Q37 (F2918 probably states the rule: gold too narrow, to validate), Q16 (company health plan: the word "apprenti" still pulls apprenticeship texts), Q57, Q58, Q65, Q68, Q69.

## 5. Next steps

1. Answer runs when the Groq quota is back: v1 + language in code against v2 + language in code, same 37 questions, same model (gpt-oss-120b); then the baseline (no documents) and the red-team set.
2. Gold sources of the evaluation lead's test set: list every article or fiche section that states the rule (candidates proposed with the passage as evidence, validated by the evaluation lead).
3. Corpus gaps: either the data lead adds durée du travail, repos, travail de nuit and médecine du travail to `ingest/scope.yaml`, or those questions become "no answer" for 3 Nov.
4. Rerun retrieval, then answers (system and baseline, same questions, 95 % intervals), then the red-team set (`eval/redteam.jsonl`, 28 cases).
