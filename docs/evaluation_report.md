# Evaluation report

Evaluation lead: Nafissatou Ndiaye. Final runs of 2 and 3 Nov. 2026, answer model gpt-oss-120b on Groq (temperature 0). Every section in full, with each failure and the commands to reproduce, in `docs/evaluation_details.md`; run files in `eval/runs/`.

## 1. Test set
`eval/testset.jsonl`: **99 questions** (78 French, 21 English), written from the point of view of apprentices, students looking for an apprenticeship and international students (paraphrases, SMS spelling), plus 11 proposed by the retrieval lead and accepted after checking. Three families:

| Family | n | Expected |
|---|---|---|
| in_corpus | 54 | the answer, citing a gold source |
| company_dependent | 26 | the general rule with its source **and** what to ask HR or the school |
| no_answer | 19 | NOT FOUND (school fees, a given company's salary, the user's own contract number) |

Included: 15 questions on the aivancity regulations and 3 immigration cases with a referral to the school (course feedback of 2 Nov.). Every gold source was checked against the corpus and every added one by a quote found word for word in the official text (`eval/review_testset.py --check`); 21 wrong or too narrow sources were corrected (`eval/testset_review.csv`).

## 2. Metrics and judge
- **Main metric: correct and cited.** On the 80 questions with an answer: the judge says the answer is correct **and** it cites a gold source (checked in code). Target 0.8.
- In code: NOT FOUND on the 19 questions without an answer; answered when answerable (the price of abstention); invented citations (an id that was not among the retrieved sources); a gold source among the 5 passages (retrieval alone); latency.
- Every rate with its **95 % Wilson interval**.
- **Why a judge.** The starter metric (reference text found word for word) gave 0/27 on answers that were right. The judge (`eval/judge.py`) is another model than the answer model (qwen/qwen3.8-27b), reads the question, the reference, the text of the gold sources and the answer, and returns correct or not, with a rubric per family.
- **Calibration before use.** 40 answers labelled by hand by the evaluation lead without seeing the judge: Cohen's kappa **0.74** (agreement 35/40), threshold 0.6 set beforehand. The first run gave 0.33; reading the disagreements showed 8 labels that had not applied our own company_dependent rule and one missing gold source. No label was changed to follow the judge.
- **Baseline:** the same model and questions, prompt only, no documents (`prompts/baseline_no_documents.txt`).

## 3. Results
| System (80 answerable, 19 without an answer) | Correct and cited (95 % CI) | Correct, judge only | NOT FOUND when no answer | Invented citations | Median latency |
|---|---|---|---|---|---|
| Baseline: model alone, no documents | 0.01 (0.00 to 0.07) | 0.19 | 0/19 | n/a | 8.2 s |
| ApprentiGuide, prompt v2 (official run, 2 Nov.) | **0.44 (0.33 to 0.55)** | 0.55 | 17/19 | 1/99 | 14.1 s |
| ApprentiGuide, prompt v3 (the demo, 3 Nov.) | **0.56 (0.45 to 0.67)** | 0.65 | 18/19 | 0/99 | 14.7 s |

| Subset, correct and cited | v2 | v3 |
|---|---|---|
| in_corpus (54) | 0.57 | 0.67 (0.53 to 0.78) |
| company_dependent (26) | 0.15 | 0.35 (0.19 to 0.54) |
| French (64) / English (16) | 0.45 / 0.38 | 0.58 / 0.50 |

A gold source is among the 5 passages for 0.86 of the questions (v2 run), 0.88 (v3 run). The referral to the school's service for international students is in 6/7 (v2) and 7/7 (v3) of the immigration answers.

**Reading.** The documents make the difference: 0.44 to 0.56 against 0.01, and 17 to 18 abstentions out of 19 against none (the model alone answers the school fees or a company's salary with invented figures); the intervals do not overlap. The **target of 0.8 is not reached.** v3 against v2, paired on the same questions: 11 gained, 1 lost (exact McNemar p ≈ 0.006), but **v3 was written after reading these failures**, so its score is optimistic until measured on unseen questions.

## 4. Failure analysis (official run, prompt v2)
The 45 answers that are not correct and cited, read one by one, each counted once:

| Type | n | Example |
|---|---|---|
| Wrong, unsupported or incomplete content | 16 | Q62 cites "L2327" (the only invented citation); Q61 invents a 15-day period |
| Correct, but cites a valid source outside the gold list | 9 | Q01 cites F2918 instead of L6222-1 (gold lists not extended after reading the answers) |
| Rule without the HR or school advice | 6 | Q60: right rule on commuting costs, no "ask HR about other modes"; **fixed by prompt v3** (the "whom to ask" line is in 23/26 company_dependent answers) |
| Judge did not see the text that proves the answer | 6 | S01: the judge reads the first 2500 characters, the sentence is at character 2806 |
| Different question answered, gold not retrieved | 3 | **Q36**: the legal 35 h when the maximum (48 h, L3121-20) was asked; **still failing with v3**, a retrieval problem |
| NOT FOUND on an answerable question | 3 | Q16 (company health plan) |
| Test set written before the school regulations | 2 | Q19, Q68: answered right from aivancity-RP-3.3.9; reference rewritten |

The judge variant that reads the relevant parts of long sources scored kappa **0.54** on the same 40 labels, under the threshold: it is not used, so the scores above keep the judge's known cut (`docs/evaluation_details.md`, section 4).

**Red team** (`docs/red_team_log.md`, 28 attacks, prompt v2): 23 passed, 2 safe but not ideal (NOT FOUND instead of correcting a false premise), 3 failed (an ambiguous question answered with a guess, twice; a yes/no legal verdict). No injection, leak of the system prompt, invented article or repeated personal data.

## 5. Cost
Free Groq plan: about 3000 tokens per answered question (rewrite of the question included), 200 000 tokens per day and per model and key, so about 60 answers per day and per key; the final runs used several keys (`app/keys.py`). Judge: about 2000 tokens per answer. Retrieval, reranking and speech-to-text run locally on the CPU (no cost). At the volume of a school (a few hundred questions a day), the free plan is not enough: a paid plan, or a local model (Qwen3 8B, 49 s per answer on a laptop).

## 6. Limits
- The test set was used during development (threshold, gold additions, prompt v3): scores are optimistic; the planned hold-out set from students was **not collected**.
- One annotator; 80 answerable questions (about ±11 points); the judge reads the first 2500 characters of each source; texts of 25 and 30 Oct. 2026.
