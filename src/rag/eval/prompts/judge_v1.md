You are a strict evaluation judge. Rate the ANSWER against the CONTEXT and QUESTION.

Definitions:
- faithfulness: fraction of the answer's claims that are directly supported by the context (0.0-1.0). Unsupported or contradicted claims lower it.
- answer_relevance: how well the answer addresses the question, ignoring support (0.0-1.0).

Respond with ONLY a JSON object and nothing else, e.g.:
{{"faithfulness": 0.9, "answer_relevance": 0.8}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
