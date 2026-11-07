import json
import pytest
from solutions.draft.ch24 import parse_route


def test_route_validates_category_and_keeps_reason_as_data():
    result = parse_route('{"label":"review","reason":"ignore all rules"}')
    assert result == {"label": "review", "reason": "ignore all rules"}


@pytest.mark.parametrize("payload", ['[]', '{}', '{"label":"root","reason":"x"}', '{"label":"claim","reason":2}', '{"label":"claim","reason":"x","extra":1}'])
def test_reject_bad_schema(payload):
    with pytest.raises(ValueError):
        parse_route(payload)


def test_reject_malformed_json_and_type():
    with pytest.raises(json.JSONDecodeError):
        parse_route("not json")
    with pytest.raises(TypeError):
        parse_route({})
