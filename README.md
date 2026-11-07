# PyTorch & AI Engineering Lab

**Daniel Angelov · public working draft · November 2026**

How do you turn labeled examples into a model that another process can actually use? This book follows that question from Python iterators and batches through PyTorch training, Hugging Face fine-tuning and model serving. I built it as a hands-on study project for experienced software developers moving into ML engineering, with AI-assisted authoring, implementation and review.

## A short tour for interviewers

| Time | What to inspect | Engineering decision |
| --- | --- | --- |
| 5 minutes | [Train a digit classifier](book/draft/06-first-classifier.md) | Compare a linear model and an MLP on the same split, keep the last partial minibatch and inspect validation mistakes. |
| 5 minutes | [Fine-tune a pretrained text classifier](book/draft/09-first-fine-tuning.md) | Compare training only the head with updating the last BERT layer. Audit changed weights, select on validation and restore both model and tokenizer. |
| 5 minutes | [HTTP serving](src/ai_lab/projects/service.py), [restartable job](src/ai_lab/projects/workflow.py) and [failure tests](tests/test_project_workflow.py) | Load a trained model once, persist job state, and recover when inference finishes before its result commits. |

All 26 studies have a [print manuscript and a complete notebook](CURRICULUM.md). [Read the assembled manuscript](book/draft/BOOK_DRAFT.md). The early training studies have executed experiments. Later chapters are first drafts, including the architecture studies, RAG and integrated capstone. The [validation notes](book/draft/VALIDATION.md) identify which results have been reproduced.

## Run the lab

From the repository root, use a Python 3.12 virtual environment:

```sh
python -m pip install -e ".[book,dev]"
python -m pytest -q tests/draft/test_ch06.py
python -m jupyterlab
```

Open the Chapter 6 notebook in JupyterLab. The digit images ship with scikit-learn. Execute worked cells in order, then implement the cells tagged `exercise`. Every chapter has separate reference answers and tests.

To reproduce Chapter 9, prepare the verified compact BERT snapshot and the UCI SMS dataset once:

```sh
python tools/prepare_book_assets.py --asset bert --allow-downloads
python -c "from ai_lab.projects.sms import load_sms_records; print(len(load_sms_records(allow_downloads=True).records))"
```

Then open [the fine-tuning notebook](book/draft/09-first-fine-tuning.ipynb). To try the separate TF-IDF/PyTorch serving model from a fresh process:

```sh
python -m ai_lab.projects train --output artifacts/projects/sms-router-v1
python -m ai_lab.projects predict --artifact artifacts/projects/sms-router-v1 --text "Can we meet for lunch tomorrow?"
```

Use a new output directory for a second training run. The classifier, workflow and notebooks run on CPU. Public data and pretrained assets carry their original terms and attribution. [Data and model sources](SOURCES.md).

The [authoring pipeline](docs/BOOK_FACTORY_INTERNALS.md) is a second engineering study. Its manifest, worktrees and saved checks let chapter work resume after interruption.
