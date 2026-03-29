# Vietnamese Handwritten OCR with TrOCR

Academic project for Vietnamese handwritten text recognition using TrOCR, ToneSpatialGate, and ToneAwareLoss.

## Repository Layout

```text
.
|- apps/
|  \- streamlit_app/        # Streamlit demo app
|- artifacts/               # Local-only models and training artifacts
|- docs/
|  |- evaluation/           # Exported metrics and figures tracked in Git
|  |- report/               # Original Vietnamese project report
|  \- slides/               # Original Vietnamese presentation deck
|- local/                   # Local-only archives and admin files
|- notebooks/
|  \- trocr_training.ipynb  # EDA, training, and evaluation notebook
|- .gitignore
|- CONTRIBUTING.md
\- README.md
```

## Main Components

- `apps/streamlit_app/app.py`: uploads an image, detects text regions with EasyOCR or PaddleOCR, and recognizes each crop with TrOCR.
- `notebooks/trocr_training.ipynb`: dataset inspection, preprocessing, fine-tuning, and evaluation workflow.
- `docs/evaluation/metrics.json`: exported test metrics for quick review without opening the notebook.

## Current Results

- CER: `2.41%`
- WER: `5.77%`
- Accuracy: `94.23%`
- Char F1: `97.80%`
- Tone Accuracy: `97.29%`

## Runtime Notes

- `EasyOCR + TrOCR` has been smoke-tested on this repository layout.
- `PaddleOCR` still requires a Python version compatible with `paddlepaddle==2.6.2`; use Python `3.10` or `3.11` if you need the PaddleOCR path.

## Model Download

The fine-tuned model is intentionally excluded from Git history because `model.safetensors` exceeds GitHub's normal file size limits.

- Expected local unpack path: `artifacts/models/best_model/`
- Optional runtime override: set `OCR_MODEL_DIR` to a different model directory
- Model package URL: `TO_BE_PUBLISHED`

After downloading the model package, extract it so that files such as `config.json`, `tokenizer.json`, and `model.safetensors` live directly under `artifacts/models/best_model/`.

## Run the Streamlit App

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r apps/streamlit_app/requirements.txt
   ```

3. Place the fine-tuned model under `artifacts/models/best_model/` or set `OCR_MODEL_DIR`.

4. Start the app:

   ```powershell
   streamlit run apps/streamlit_app/app.py
   ```

## Git Workflow

- Default branch: `main`
- Feature branch prefix: `codex/`

See `CONTRIBUTING.md` for the lightweight branch and commit conventions used in this repository.

## Publishing Notes

- `artifacts/` and `local/` are split out so the public repository stays focused on code, notebooks, and documentation.
- The original Vietnamese report and slide assets are preserved under `docs/`.
- The repository slug should be ASCII-only: `vietnamese-handwritten-ocr-trocr`
