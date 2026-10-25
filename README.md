# <Product name>

> One sentence: who it helps, with what, and how.

## 1. User and problem
- **User:** <a specific person and situation>
- **Problem today:** <what is slow, costly or error-prone; how it is done now>
- **Value if it works:** <minutes saved, errors avoided, access gained>

## 2. What the system does
- **Input:** <documents, questions, audio, images…>
- **Output:** <answer with citations, JSON table, summary, audio…>
- **Technique:** <prompting / RAG / LoRA / modality>, model: <name, size, quantisation>, why this choice: <…>

## 3. How to run it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # paste your free Groq key (console.groq.com/keys)
# local instead (sensitive data, offline): ollama pull llama3.2, then switch config.yaml
python eval/eval.py             # runs the test set and writes eval/report.json
streamlit run app/ui.py         # demo interface
```

## 4. Data
See `data/README.md`: sources, licences, personal data, how the data was prepared.

## 5. Evaluation (summary)
| System | Main metric | NOT FOUND rate | Median latency |
|---|---|---|---|
| Baseline (prompt only) | | | |
| Our system | | | |

Full report: `docs/evaluation_report.md`. System card: `docs/system_card.md`.

## 6. AI tools used to build this project
| Tool | Used for |
|---|---|
| <e.g. Copilot> | <e.g. Streamlit interface> |

## 7. Team
| Member | Main role | Also worked on |
|---|---|---|
| | data | |
| | retrieval / model | |
| | evaluation and safety | |
| | product and demo | |
