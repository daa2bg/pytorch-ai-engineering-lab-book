"""Local HTTP adapter for the saved SMS model, with no training or downloads.

Run ``python -m ai_lab.projects.service --bundle PATH --port 8000`` for a
loopback-only demonstration. Use ``with TestClient(create_app(PATH))`` in tests
so startup/shutdown actually run. A corrupt/missing bundle fails startup.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from .sms import TASK_VERSION, load_sms_artifact, predict_sms

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=128,
                                      pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """No coercion, unknown fields, artifact paths, or request-selected policy."""
    model_config = ConfigDict(extra="forbid")
    request_id: Identifier
    text: Annotated[StrictStr, Field(min_length=1, max_length=4096)]

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class Probabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ham: Probability
    spam: Probability


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: Identifier
    task_id: Literal["sms-routing.v1"]
    artifact_id: Identifier
    label: Literal["ham", "spam"]
    probabilities: Probabilities
    confidence: Probability
    route: Literal["deliver", "quarantine", "review"]
    policy_version: Identifier

    @model_validator(mode="after")
    def consistent_prediction(self):
        p = self.probabilities
        expected_label = "spam" if p.spam > p.ham else "ham"
        expected_confidence = p.spam if self.label == "spam" else p.ham
        expected_route = "quarantine" if self.label == "spam" else "deliver"
        if (abs(p.ham + p.spam - 1) > 1e-5 or self.label != expected_label or
                abs(self.confidence - expected_confidence) > 1e-6 or
                self.route not in {"review", expected_route}):
            raise ValueError("Inconsistent model prediction")
        return self


def validate_request(request: dict) -> dict:
    """The shared HTTP/workflow input contract; returns a fresh dictionary."""
    return PredictionRequest.model_validate(request).model_dump()


def validate_prediction(response: dict, *, request_id, artifact_id, policy_version) -> dict:
    """Reject malformed outputs and a predictor silently changing model identity."""
    result = PredictionResponse.model_validate(response).model_dump()
    if (result["request_id"] != request_id or result["artifact_id"] != artifact_id or
            result["policy_version"] != policy_version):
        raise ValueError("Prediction request/model/policy identity mismatch")
    return result


class SavedPredictor:
    """A small synchronous callable shared by HTTP and durable inference jobs.

    Construct through load_predictor. Properties identify the checked bundle;
    __call__ accepts one request dict and returns the full HTTP response dict.
    The lock serializes local model calls; this is not a batch scheduler.
    """
    def __init__(self, artifact):
        self._artifact = artifact
        self._lock = Lock()

    @property
    def artifact_id(self):
        return self._artifact.metadata["artifact_id"]

    @property
    def policy_version(self):
        return self._artifact.metadata["routing_policy"]["version"]

    def __call__(self, request):
        request = validate_request(request)
        with self._lock:
            predicted = predict_sms(self._artifact, [request["text"]])[0]
        response = {"request_id": request["request_id"], **predicted}
        # label_id belongs to the lower-level model API, not the HTTP contract.
        response.pop("label_id")
        return validate_prediction(response, request_id=request["request_id"],
                                   artifact_id=self.artifact_id, policy_version=self.policy_version)


def load_predictor(bundle_dir) -> SavedPredictor:
    """Restore a trusted local bundle once, without data access or retraining."""
    return SavedPredictor(load_sms_artifact(bundle_dir))


def create_app(bundle_dir) -> FastAPI:
    """Create an app; model loading happens exactly once per lifespan startup."""
    bundle = Path(bundle_dir).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = load_predictor(bundle)
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="SMS routing teaching service", version="1", lifespan=lifespan)
    app.state.predictor = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _error):
        # Pydantic's default details may echo submitted message text. Return a
        # stable category, not raw input or exception strings, at this boundary.
        return JSONResponse(status_code=422, content={"error": "invalid_request"})

    def ready_predictor():
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(503, detail={"error": "not_ready"})
        return predictor

    @app.get("/health")
    def health():
        predictor = ready_predictor()
        return {"ready": True, "schema_version": 1, "task_id": TASK_VERSION,
                "artifact_id": predictor.artifact_id, "policy_version": predictor.policy_version}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        predictor = ready_predictor()
        try:
            return predictor(request.model_dump())
        except (TimeoutError, ConnectionError) as exc:
            raise HTTPException(503, detail={"error": "prediction_unavailable"},
                                headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            raise HTTPException(500, detail={"error": "prediction_failed"}) from exc

    return app


def main(argv=None):
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Serve one saved SMS model on loopback only")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    uvicorn.run(create_app(args.bundle), host="127.0.0.1", port=args.port,
                access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
