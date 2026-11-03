# Evaluation report: details

Appendix of `docs/evaluation_report.md` (the 2-page report): every section in full, with the official run of 2 Nov. (prompt v2). The prompt v3 run of 3 Nov. is in the report and in `docs/decisions_and_experiments.md` (E21).

Evaluation lead: Nafissatou Ndiaye. Scores from the final runs of 2 Nov. (commands in section 7).

## 1. Test set
- **Main set** (`eval/testset.jsonl`): 99 questions, 78 in French and 21 in English, in three families:
  - **in_corpus** (54): the answer is in the texts; expected an answer with a citation of a gold source.
  - **company_dependent** (26): the law gives a general rule, the rest depends on the employer or the school; expected the rule with its source **and** what to ask HR or the school.
  - **no_answer** (19): school calendar and fees, a given company's salary, the user's own contract number; expected NOT FOUND.
- **Added on 2 Nov.** (after the professor's review and the school regulations):
  - **School regulations** (S01 to S15): 11 answered by the aivancity rules (absences of apprentices, exams, resits, approval of the apprenticeship, months in a company, gap year), 2 school and employer (absences deducted from pay, a working day on a class day), 2 still without an answer (date of the resit session, name of the tutor). Q41, Q43 and Q44, written as `no_answer` before the regulations were in the corpus, are now answered by them (pedagogical rules 3.3.1, 3.3.2, 3.3.8) and were moved to the matching family. Every quote was checked word for word in the PDF text (`python eval/review_testset.py --check`, needs the PDF files in `data/private/`).
  - **Immigration** (Q71 to Q73, CESEDA: working too many hours on a student card, staying to look for a job after a master's degree, a card that expires during the contract). For these and Q19, Q66, Q67, A04 (`refer_international` in the file), the answer must give the rule **and** send the student to the school's service for international students (wording of the run of 2 Nov.: "bureau international", changed after the run because aivancity has no office of that name). The referral is added in code when the answer cites a source about foreign students (`international_referral` in `app/pipeline.py`) and is checked in code (`referred_international` in the report).
- **How it was built.** 70 questions written by the evaluation lead from the point of view of apprentices, job seekers and international students (paraphrases, SMS spelling, English), plus 11 proposed by the retrieval lead (A01 to A11) and accepted. Key case: Q19, *"I am a first-year international student in the PGE. Can I do an apprenticeship this year?"*.
- **Review of the gold sources** (`eval/testset_review.csv`, applied by `eval/review_testset.py`). Every gold source was checked against the corpus: 21 of the lead's sources were wrong or too narrow (for example Q26 cited L6222-34, about sitting the exams, instead of L6222-24) and were corrected; 38 sources proposed by the retrieval lead were accepted after checking their quote word for word in the official text; placeholders were replaced; reference answers were rewritten from the texts (figures, durations, conditions). One proposal was refused (Q20 keeps L6222-24, which does answer the question).
- **Hold-out set** (`eval/testset_holdout.jsonl`): planned, **not collected** before the demo (questions from students through an anonymous form, written down before any run and never used to tune the system). Without it, every score of this report is on a set used during development; see section 6.

## 2. Metrics
| Metric | Definition | Target |
|---|---|---|
| **Correct and cited** (main) | On questions with an answer: the judge says the answer is correct, **and** it cites at least one gold source (checked in code). | 0.80 |
| **NOT FOUND rate** | On questions without an answer: the system abstains (checked in code). | ≥ 0.80 |
| Answered when answerable | On questions with an answer: no NOT FOUND (the price of abstention, "false refusals"). | high |
| Invented citations | Answers citing an id that was not among the retrieved sources. | 0 |
| Recall@5 | A gold source is among the 5 passages given to the model (retrieval, no LLM). | |
| Median latency | Seconds per answer. | |

Every rate is given with its **95 % Wilson interval**: 16/20 is "between 58 % and 92 %", not "80 %". This is why the set has 99 questions and not 20.

**Why a judge.** The starter metric `contains_reference` checks whether the reference answer appears word for word in the answer: on the two development answer runs it scored **0/27 and 0/28** (95 % CI 0 to 0.13), while the same answers cited a gold source 0.70 and 0.68 of the time; A03, which gives the right "cinq jours ouvrables", is marked wrong. It measures wording, not correctness, and is kept in the report only for comparison.

**The judge** (`eval/judge.py`). An LLM that is not the answer model (`qwen/qwen3.8-27b` on Groq; an LLM tends to prefer its own answers; `llama-3.3-70b-versatile`, planned first, is no longer served by Groq, and the only other chat models left are gpt-oss) reads the question, the reference answer, **the official text of the gold sources** and the answer, and returns `{"reason", "correct"}` at temperature 0. The rubric differs by family (key figure and condition for in_corpus; rule *and* HR advice for company_dependent). It does not judge citations (done in code) or NOT FOUND (done in code): the judge is used only where a rule cannot decide.

**Calibration.** 40 answers from earlier runs (20 from gpt-oss-120b with prompt v1, 20 from Qwen3 8B with prompt v2), shuffled, labelled by hand by the evaluation lead without seeing the judge's verdict (`eval/labels/to_label.csv` → `eval/labels/labels.csv`; result in `eval/labels/labels_calibration.json`). Cohen's kappa, ≥ 0.6 required before using the judge:

| Step | Agreement | Kappa |
|---|---|---|
| First run | 26/40 (0.65) | 0.33 |
| After the two changes below | 35/40 (0.875) | **0.74** |

Reading the 14 first disagreements one by one showed two problems on our side, not the judge's:
- **The rule for company_dependent questions was not applied in the labels.** The test set requires the general rule *and* what to ask HR or the school; 8 answers without the HR advice had been labelled correct (Q20, Q59 twice, Q60, Q64, Q66 twice, Q69). The evaluation lead confirmed the rule and relabelled them 0 (`human_note` in the file). Labels were changed only where the rule written before the calibration decided; no label was changed to follow the judge.
- **A gold source was missing:** for Q01 the judge did not have D6222-1, the article with the exceptions up to 35 years, and marked a correct answer wrong. D6222-1 was added to the gold sources (`eval/testset_review.csv`).

The 5 disagreements left were kept as labelled:

| Id | Human | Judge | What the judge says |
|---|---|---|---|
| Q08 | incorrect | correct | right percentage (43 %), the human found the answer incomplete |
| Q29 | incorrect | correct | right percentage (61 %), the "or the conventional minimum wage" condition is missing |
| Q05 (Qwen) | incorrect | correct | at least 50 % of the pass, stated right |
| Q39 (Qwen) | correct | incorrect | an "aide forfaitaire de 1 000 €" that is in none of the texts (checked) |
| Q31 | correct | incorrect | night-work derogations mixed with the working-time derogations of F2216 (checked) |

The judge is a little stricter on invented details and a little more lenient on incomplete answers. Some of its reasons are wrong even when its verdict agrees (it called L6223-8 "non-existent" when the article was only absent from the texts it was given), which is why its verdicts are counted but its reasons are not used as proof. Limits: one annotator; 20 of the 40 answers come from a Qwen model, the judge's family (the final answer model is gpt-oss).

## 3. Baseline vs system
Final runs of 2 Nov. (`eval/runs/final_system.json`, `eval/runs/final_baseline.json`): same answer model (gpt-oss-120b on Groq, temperature 0), same 99 questions, same judge. The baseline is the model with a prompt and no documents (`prompts/baseline_no_documents.txt`, D30). 80 questions have an answer, 19 do not; no question was left out (`corpus_gaps` empty: the regulation files were loaded).

| System | Correct and cited (95 % CI) | Correct, judge only | NOT FOUND when no answer | Answered when answerable | Invented citations | Median latency |
|---|---|---|---|---|---|---|
| Baseline: model alone, no documents | 0.01 (0.00 to 0.07), 1/80 | 0.19 (0.12 to 0.29) | 0.00 (0 to 0.17), 0/19 | 1.00 | n/a (no sources to cite) | 8.2 s |
| ApprentiGuide (hybrid retrieval + reranker + abstention, prompt v2) | **0.44 (0.33 to 0.55), 35/80** | 0.55 (0.44 to 0.65) | **0.89 (0.69 to 0.97), 17/19** | 0.96 (0.90 to 0.99) | 1/99 (Q62) | 14.1 s |
| Same, hold-out set | not run (no hold-out set collected) | | | | | |

Retrieval and citations of the same run, checked in code: a gold source among the 5 passages **0.86** (0.77 to 0.92); a gold source cited 0.69 (0.58 to 0.78). Immigration questions: the referral to the school's international service is in 6 of 7 answers (Q19 missing, see section 4).

| Subset (system) | n | Correct and cited | Correct, judge only | Baseline, judge only |
|---|---|---|---|---|
| in_corpus | 54 | 0.57 (0.44 to 0.70) | 0.72 (0.59 to 0.82) | 0.09 |
| company_dependent | 26 | 0.15 (0.06 to 0.34) | 0.19 (0.09 to 0.38) | 0.38 |
| French | 64 | 0.45 (0.34 to 0.57) | 0.58 | 0.20 |
| English | 16 | 0.38 (0.19 to 0.61) | 0.44 | 0.13 |
| School regulations (S) | 13 | 0.46 (0.23 to 0.71) | 0.54 | 0.00 |

**What this says.** Against the model alone, the documents make the difference on everything that can be checked: 0.44 against 0.01 correct and cited, 0.55 against 0.19 correct for the judge (the intervals do not overlap), and 17 of 19 abstentions against none (the model alone answers every question about the school calendar, fees or a company's salary, often with an invented figure). The target of 0.8 is **not reached**. The weak point is the `company_dependent` family (0.15): the rule is often right, the "ask your HR / school" part is missing, and the model alone, which says "ask your HR" by default, does better than the system on this family (0.38 against 0.19 for the judge). Read the scores with section 4: part of the failures are measurement errors (the judge did not see the whole text, a correct source that is not in the gold list).

Development measurements by the retrieval lead (`docs/decisions_and_experiments.md`), for context only (made on this test set while tuning, with gold sources not yet validated): Recall@5 0.87 (0.76 to 0.93) on 61 questions; cited gold 0.70 (0.52 to 0.84) and 0 invented citations on 37 answers; NOT FOUND 8/9. The final run gives the same retrieval figure on 80 questions (0.86).

## 4. Error analysis
The 45 system answers that are not "correct and cited", read one by one against the texts, each counted once under its main type:

| Type | Count | Example |
|---|---|---|
| **Wrong, unsupported or incomplete content** | 16 | Q61 (an invented 15-day period, D3142-16), Q62 (cites "L2327", the only invented citation), Q72 (L422-14 and "within four years"), S04 (a "48 hours" deadline from F303 that the school rules do not set), Q31 (night-work hours of minors missing). Some of these verdicts may also come from the cut text below; the re-judging will tell. |
| **Correct, but cites a valid source that is not in the gold list** | 9 | Q01: right ages and exceptions, cites F2918 (the Service-Public page on the apprentice contract) and not L6222-1. Gold lists were kept as written before the run: adding sources after reading the answers would raise the score without a new measurement. |
| **Rule without the HR / school advice** (company_dependent) | 6 | Q60 (commute): right rule, cites L3261-2, no "ask HR about other modes"; also Q44, Q59, Q64, Q69, Q70. The prompt does not ask for this part, and it is why the family scores 0.15. |
| **Judge did not see the text that proves the answer** | 6, confirmed by reading | S01: the answer "no tolerance of absence for apprentices" is right (aivancity-RP-4.1.4), but the judge cut each source at 2500 characters and the sentence is at character 2806; same for S12 (deduction from pay, character 2938), Q39 (aid amounts at character 3487 of F23556), Q02, Q29, A01 (F2918 is 32 000 characters). 48 of the 80 answers have a gold source longer than 2500 characters. Fixed after the run, see below. |
| **Different question answered, gold not retrieved** | 3 | Q03 and Q33 (trial period: L6222-18 not among the 5 passages, the answer gives the period after the contract), Q36 (35 h legal duration when the maximum 48 h was asked). |
| **NOT FOUND on an answerable question** | 3 | Q16 (company health plan), Q41, Q57. The other way, Q14 and Q53 were answered when no answer was expected (the 2 of 19 in section 3). |
| **Test set written before the school regulations** | 2 | Q19 (the key case) and Q68: the system answers "not in first year, the PGE has apprenticeship in years 4 and 5" (aivancity-RP-3.3.9), which is right for aivancity; the reference answers did not know this rule. Found on reading the final run; both now have RP-3.3.9 as gold source and a rewritten reference (`testset_review.csv`, note in `review_note`). Q19 also lacks the referral, because the referral follows the cited source and RP-3.3.9 is not an immigration text. |

**Changes after the run.** Two of these are measurement errors, not system errors:
- **The judge** reads only the first 2500 characters of each gold source. A variant that reads the most relevant chunks of each source, up to 6000 characters, was written (`relevant_text` in `eval/judge.py`, test in `tests/test_eval_tools.py`) and calibrated on the same 40 labels before any use: **kappa 0.54**, under the 0.6 threshold set beforehand. It is **not used** (off by default, `JUDGE_RELEVANT_PARTS=1`); every score of this report comes from the calibrated judge (kappa 0.74), cut included.
- **Q19 and Q68**, as said in the table.

What was changed next: prompt v3, written from this error analysis ("decide whether the answer depends on the employer or the school, then end with whom to ask and what"), measured on the same 99 questions with the same judge: correct and cited 0.56 (0.45 to 0.67) against 0.44, company-dependent 9/26 against 4/26 (`docs/decisions_and_experiments.md`, E21). Still to do: retrieval of L6222-18 and L3121-20 (trial period, maximum working time).

## 5. Cost and latency
Median **14.1 s** per answer for the system (90th percentile 23 s), 8.2 s for the model alone: retrieval, reranking on the CPU and the rewrite of the question (an extra model call) cost about 6 s. Free Groq plan: about 3000 tokens per answered question, 200 000 tokens per day and per model, so about 60 full answers per day and per key; the final run of 99 questions needed several keys (key rotation, `app/keys.py`). Judge: about 2000 to 4000 tokens per answer (longer since it reads more of each source). At the expected volume of a school (a few hundred questions a day), the free plan is not enough: a paid plan, or a smaller model served locally for the answer step.

## 6. Limits of this evaluation and how we handled them
- **The abstention threshold and the gold additions were tuned on this test set**, and gold sources were added after reading what the system retrieved: both make the scores optimistic. Answer: the **hold-out set**, collected from students and fixed before any run, and the `--lead-gold-only` option (gold sources set before seeing results).
- **One annotator** for the calibration labels: a second labeller on 20 answers would measure human agreement too.
- **Small sets**: differences inside the confidence intervals are not conclusions.
- **The school regulations were added to the corpus on 2 Nov. (73 chunks, see `data/README.md`), after the `no_answer` questions were written.** They were read again against the regulation chunks: Q41, Q43 and Q44 are now answered by the rules and changed family (note in `review_note`); Q11, Q12 and Q52 stay `no_answer` (no deadline, fees or change of rhythm in the rules). Runs need the PDF files in `data/private/`: without them the S questions and Q41, Q43, Q44 have no gold source in the database and are listed in `corpus_gaps`, not scored.
- **Snapshot**: texts of 25 and 30 Oct. 2026; a faithful answer can become wrong when the law changes.

## 7. How to reproduce
```bash
python eval/review_testset.py --check                                  # quotes of the gold sources still in the texts
python eval/judge.py calibrate eval/labels/labels.csv                  # judge vs human labels
python eval/eval.py --model openai/gpt-oss-120b --prompt prompts/system_v2.txt --judge --out eval/runs/final_system.json
python eval/eval.py --model openai/gpt-oss-120b --prompt prompts/baseline_no_documents.txt --baseline --judge --out eval/runs/final_baseline.json
python eval/judge.py rejudge eval/runs/final_system.json      # after a new calibration: verdicts with the fixed judge
python eval/judge.py rejudge eval/runs/final_baseline.json
python eval/eval.py --model openai/gpt-oss-120b --prompt prompts/system_v2.txt --judge --testset eval/testset_holdout.jsonl --out eval/runs/final_holdout.json   # once the hold-out set is collected
python eval/redteam.py --model openai/gpt-oss-120b
```

## 8. What we would do next
Second annotator; larger hold-out set from real users of the demo; retrieval of "same rights as other employees" for company-dependent questions; an agent that asks the user their situation (age, year, nationality) before answering.
