"""Reference answer: validated prediction boundary, independent of HTTP."""
def predict_request(payload: dict, predictor, *, max_chars: int = 200) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    if set(payload) != {"text"}:
        raise ValueError("payload must contain exactly text")
    text = payload["text"]
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip() or len(text) > max_chars:
        raise ValueError("text must be nonblank and within max_chars")
    label = predictor(text)
    if not isinstance(label, str) or label not in {"claim", "invoice", "general"}:
        raise ValueError("predictor returned an unknown label")
    return {"label": label}
