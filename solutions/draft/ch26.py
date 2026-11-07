"""Reference validation boundary for the toy document capstone."""


def validate_records(records):
    output = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dictionary")
        if not {"id", "text", "label"} <= record.keys():
            raise ValueError("missing required field")
        if any(not isinstance(record[key], str) for key in ("id", "text", "label")):
            raise TypeError("id, text, and label must be strings")
        identity, text = record["id"].strip(), record["text"].strip()
        if not identity or not text or record["label"] not in {"claim", "invoice"}:
            raise ValueError("invalid id, text, or label")
        if identity in seen:
            raise ValueError("duplicate document id")
        seen.add(identity)
        output.append({"id": identity, "text": text, "label": record["label"]})
    return output
