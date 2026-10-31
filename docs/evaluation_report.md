# Evaluation report

Evaluation lead: Nafissatou Ndiaye. Values marked **TBD** are filled from the final runs (commands in section 7).

## 1. Test set
- **Main set** (`eval/testset.jsonl`): 81 questions, 64 in French and 17 in English, in three families:
  - **in_corpus** (39): the answer is in the texts; expected an answer with a citation of a gold source.
  - **company_dependent** (22): the law gives a general rule, the rest depends on the employer or the school; expected the rule with its source **and** what to ask HR or the school.
  - **no_answer** (20): school calendar, a given company's salary, the user's own contract number; expected NOT FOUND.
- **How it was built.** 70 questions written by the evaluation lead from the point of view of apprentices, job seekers and international students (paraphrases, SMS spelling, English), plus 11 proposed by the retrieval lead (A01 to A11) and accepted. Key case: Q19, *"I am a first-year international student in the PGE. Can I do an apprenticeship this year?"*.
- **Review of the gold sources** (`eval/testset_review.csv`, applied by `eval/review_testset.py`). Every gold source was checked against the corpus: 21 of the lead's sources were wrong or too narrow (for example Q26 cited L6222-34, about sitting the exams, instead of L6222-24) and were corrected; 38 sources proposed by the retrieval lead were accepted after checking their quote word for word in the official text; placeholders were replaced; reference answers were rewritten from the texts (figures, durations, conditions). One proposal was refused (Q20 keeps L6222-24, which does answer the question).
- **Hold-out set** (`eval/testset_holdout.jsonl`, **TBD**): questions collected from students through an anonymous form, written down before any run and never used to tune the system. See section 6.

## 2. Metrics
| Metric | Definition | Target |
|---|---|---|
| **Correct and cited** (main) | On questions with an answer: the judge says the answer is correct, **and** it cites at least one gold source (checked in code). | 0.80 |
| **NOT FOUND rate** | On questions without an answer: the system abstains (checked in code). | ≥ 0.80 |
| Answered when answerable | On questions with an answer: no NOT FOUND (the price of abstention, "false refusals"). | high |
| Invented citations | Answers citing an id that was not among the retrieved sources. | 0 |
| Recall@5 | A gold source is among the 5 passages given to the model (retrieval, no LLM). | |
| Median latency | Seconds per answer. | |

Every rate is given with its **95 % Wilson interval**: 16/20 is "between 58 % and 92 %", not "80 %". This is why the set has 81 questions and not 20.

**Why a judge.** The starter metric `contains_reference` checks whether the reference answer appears word for word in the answer: on the two development answer runs it scored **0/27 and 0/28** (95 % CI 0 to 0.13), while the same answers cited a gold source 0.70 and 0.68 of the time; A03, which gives the right "cinq jours ouvrables", is marked wrong. It measures wording, not correctness, and is kept in the report only for comparison.

**The judge** (`eval/judge.py`). An LLM that is not the answer model (`llama-3.3-70b-versatile` on Groq; an LLM tends to prefer its own answers) reads the question, the reference answer, **the official text of the gold sources** and the answer, and returns `{"reason", "correct"}` at temperature 0. The rubric differs by family (key figure and condition for in_corpus; rule *and* HR advice for company_dependent). It does not judge citations (done in code) or NOT FOUND (done in code): the judge is used only where a rule cannot decide.

**Calibration.** 40 answers from earlier runs (two models), shuffled, labelled by hand by the evaluation lead without seeing the judge's verdict (`eval/labels/to_label.csv`). Agreement judge vs human: **TBD** (Cohen's kappa; ≥ 0.6 required before using the judge). Disagreements read one by one: **TBD**.

## 3. Baseline vs system
Same answer model (gpt-oss-120b on Groq), same questions, same judge.

| System | Correct and cited (95 % CI) | Correct, judge only | NOT FOUND when no answer | Answered when answerable | Invented citations | Median latency |
|---|---|---|---|---|---|---|
| Baseline: model alone, no documents | TBD (no citation possible) | TBD | TBD | TBD | n/a | TBD |
| ApprentiGuide (hybrid retrieval + reranker + abstention, prompt v2) | **TBD** | TBD | TBD | TBD | TBD | TBD |
| Same, hold-out set | TBD | TBD | TBD | TBD | TBD | TBD |

Per family and per language: **TBD** (`by_tag` in the report).

Development measurements by the retrieval lead (`docs/decisions_and_experiments.md`), for context only (made on this test set while tuning, with gold sources not yet validated): Recall@5 0.87 (0.76 to 0.93) on 61 questions; cited gold 0.70 (0.52 to 0.84) and 0 invented citations on 37 answers; NOT FOUND 8/9.

## 4. Error analysis
**TBD** from the final run: count per type, one example each. Types already seen in development:
- **Company-dependent answered NOT FOUND** (Q58 bonus, Q16 company health plan): the source that states "same rights as the other employees" is not retrieved.
- **Different question answered** (Q36: legal 35 h given when the maximum was asked).
- **Language** (English questions answered in French): fixed in code (D14), to be checked.
- **Wrong abstention** (a question about the user's own contract number found F2918 with a high score).

## 5. Cost and latency
Free Groq plan: about 3000 tokens per answered question, 200 000 tokens per day and per model, so about 60 full answers per day and per key. Judge: about 2000 tokens per answer. **TBD**: median latency of the final run, cost at expected volume.

## 6. Limits of this evaluation and how we handled them
- **The abstention threshold and the gold additions were tuned on this test set**, and gold sources were added after reading what the system retrieved: both make the scores optimistic. Answer: the **hold-out set**, collected from students and fixed before any run, and the `--lead-gold-only` option (gold sources set before seeing results).
- **One annotator** for the calibration labels: a second labeller on 20 answers would measure human agreement too.
- **Small sets**: differences inside the confidence intervals are not conclusions.
- **Snapshot**: texts of 25 and 30 Oct. 2026; a faithful answer can become wrong when the law changes.

## 7. How to reproduce
```bash
python eval/review_testset.py --check                                  # quotes of the gold sources still in the texts
python eval/judge.py calibrate eval/labels/labels.csv                  # judge vs human labels
python eval/eval.py --model openai/gpt-oss-120b --prompt prompts/system_v2.txt --judge --out eval/runs/final_system.json
python eval/eval.py --model openai/gpt-oss-120b --baseline --judge --out eval/runs/final_baseline.json
python eval/eval.py --model openai/gpt-oss-120b --prompt prompts/system_v2.txt --judge --testset eval/testset_holdout.jsonl --out eval/runs/final_holdout.json
python eval/redteam.py --model openai/gpt-oss-120b
```

## 8. What we would do next
Second annotator; larger hold-out set from real users of the demo; retrieval of "same rights as other employees" for company-dependent questions; an agent that asks the user their situation (age, year, nationality) before answering.
