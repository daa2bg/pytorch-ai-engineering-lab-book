# Study 14 — Build an evidence-first RAG workflow

**First draft.** Connect chunking, retrieval, source citations, and abstention in an offline workflow. Prerequisites: vector retrieval and relevance fixtures. The response stage here is a deterministic extractive baseline, **not an LLM**. Its limitations are part of the study.

## Retrieval and generation solve different problems

**Retrieval-augmented generation (RAG)** supplies retrieved evidence to a generator before it answers. Retrieval chooses candidate material; generation decides how to use it. A system can retrieve poorly and sound fluent, or retrieve the right material and still produce an unsupported answer.

Start with inspectable intermediate results. Each source document gets a stable ID. **Chunking** breaks it into smaller passages while retaining source references. Overlap can preserve context across boundaries, but it creates duplicate information and additional index entries. Choose chunk size based on the downstream representation and task rather than assuming a universal number.

For this small lab, chunk by whitespace words. Model token counts can differ, so this is not a substitute for measuring the actual tokenizer's input length.

## Keep provenance while chunking

```python
sources = {
    "billing": "Invoice payments are due within thirty days. Contact billing for a receipt.",
    "claims": "Vehicle claims need a repair estimate. Submit the estimate with your claim.",
}

def chunk_words(source_id, text, *, size=7, overlap=2):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > overlap >= 0")
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append({"id": f"{source_id}:{start}", "source": source_id,
                       "text": " ".join(words[start:start + size])})
        if start + size >= len(words):
            break
    return chunks

chunks = []
for source_id, text in sources.items():
    chunks.extend(chunk_words(source_id, text))
assert all(chunk["source"] in sources for chunk in chunks)
assert len({chunk["id"] for chunk in chunks}) == len(chunks)
print(chunks[0])
```

The chunk ID includes its source and starting word offset. If a document changes, a production system also needs a source version; otherwise the same ID might refer to different text after reindexing. Keep the chunk text and source version together.

## Retrieve before answering

To keep the response behavior transparent, rank by token overlap rather than introducing another model. This lexical method misses synonyms and can overvalue generic words. The goal is a visible contract, not competitive search quality.

```python
def words(text):
    return set(text.lower().strip(".?!").split())

def retrieve(question, passages, *, k=2):
    query_words = words(question)
    scored = [(len(query_words & words(p["text"])), p) for p in passages]
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [passage for score, passage in scored[:k] if score > 0]

evidence = retrieve("invoice payments", chunks)
assert evidence
assert evidence[0]["source"] == "billing"
assert retrieve("penguin habitat", chunks) == []
print("Retrieved IDs:", [item["id"] for item in evidence])
```

The filtering policy refuses zero-overlap passages. That is better than returning an arbitrary first chunk, but a positive overlap is still not proof that the passage answers the question. A mature system needs stronger retrieval evaluation and calibrated answer policies.

## An extractive response makes support visible

```python
def extractive_response(question, passages):
    evidence = retrieve(question, passages)
    if not evidence:
        return {"answer": None, "citations": [], "status": "abstain"}
    selected = evidence[0]
    return {"answer": selected["text"], "citations": [selected["id"]],
            "status": "extractive"}

response = extractive_response("invoice payments", chunks)
assert response["status"] == "extractive"
cited = next(p for p in chunks if p["id"] == response["citations"][0])
assert response["answer"] == cited["text"]
assert extractive_response("penguin habitat", chunks)["status"] == "abstain"
print(response)
```

This copies evidence verbatim. It is easy to verify the citation, but it does not synthesize an answer or guarantee relevance. If you replace it with an LLM, pass bounded evidence, preserve citation IDs, and treat retrieved text as data rather than trusted instructions. A source saying “ignore the system rules” must not become an instruction to the application.

Separate evaluation questions: did retrieval include the known relevant source? Did the response use only supported facts? Did the system abstain when evidence was absent? Do not collapse those questions into a single impression of answer quality.

## Essential exercise — measure retrieval coverage

Implement `recall_at_k(retrieved_ids, relevant_ids, k)`. Divide the number of unique relevant IDs recovered in the first `k` positions by the number of unique relevant IDs. Duplicate retrieved IDs receive no extra credit. Reject an empty relevance set or nonpositive `k` with `ValueError`. The caller supplies ID sequences and an integer. Empty retrieved results return zero.

```python exercise
def recall_at_k(retrieved_ids, relevant_ids, k):
    raise NotImplementedError("Count recovered relevant IDs in the first k positions")

assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.5
assert recall_at_k(["wrong", "b", "a"], ["a", "b"], 3) == 1.0
assert recall_at_k([], ["a"], 2) == 0.0
```

**Optional study:** replace the extractive baseline with a local generator. Keep the retrieval fixture unchanged, add unsupported-answer and no-evidence cases, and report generation latency separately. A fluent answer is not itself evidence of grounding.

**Completion gate:** retain source IDs through chunking, demonstrate a supported extract and an abstention, then calculate retrieval recall independently of response wording.

Primary implementation study: https://huggingface.co/learn/cookbook/en/advanced_rag . Reference answer: `solutions/draft/ch14.py`; tests: `tests/draft/test_ch14.py`. No hosted model or actual LLM generation is executed here.
