# Agent sheet: ApprentiGuide

From an assistant that answers to an agent that acts. The Generative AI product (cited answers from the Labour Code, the CESEDA, Service-Public.fr and the school regulations, with NOT FOUND when the texts do not answer) is the base of the agent built in the Agentic AI course. This sheet is completed after each session; the deliverables are listed at the end.

## User and goal

- **User:** apprentices, students looking for an apprenticeship and international students of the school, such as Inès (22, international student, starting year 4 of the PGE).
- **What the agent adds:** the Generative AI product cites the rule but cannot act on the user's situation. It does not know today's date, cannot compute a deadline or an amount, guesses when a question is vague (our main measured weakness), and cannot prepare a message to HR or the school.

## S1. Five real requests: workflow or agent?

A workflow follows steps fixed in code; an agent decides at each step which tool to call, from what it has observed. We keep a workflow whenever the steps are always the same: it is cheaper, faster and easier to test.

| # | Request (as a user would write it) | Decision | Why |
|---|---|---|---|
| 1 | « Combien de jours de congés j'ai pour réviser mes examens ? » | **Workflow** | One search and one cited answer: the steps are always the same, which is exactly our current RAG pipeline. |
| 2 | « J'ai raté mon examen mardi à cause d'un train en retard. Est-ce que je peux encore le justifier, et à qui ? » | **Agent** | It needs today's date, the school rule and a deadline computation, and which tools to call next depends on what the rule says. |
| 3 | « J'ai le droit à combien ? » | **Agent** | The request is vague, so the agent must first ask what the user means instead of guessing, which our RAG cannot do. |
| 4 | « J'ai 22 ans et je suis en deuxième année d'apprentissage : combien je vais gagner par mois, en euros ? » | **Workflow** | The steps are fixed: find the pay rule, then compute the amount with a calculation tool, never by the model. |
| 5 | « Prépare un email à mes RH pour poser mes 5 jours de congé d'examen avant ma session de juin. » | **Agent** | It combines the rule, the user's dates and a draft email, and sending it is a write action that must wait for the user's approval. |
