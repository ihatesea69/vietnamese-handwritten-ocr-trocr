# Contributing

## Branching

- Stable branch: `main`
- Feature branch prefix: `codex/`

Examples:

- `codex/readme-cleanup`
- `codex/paddleocr-compat`
- `codex/model-download-docs`

## Commits

- Keep commits scoped to one logical change.
- Prefer imperative commit messages, for example:
  - `Initialize GitHub-ready project structure for Vietnamese Handwritten OCR`
  - `Document external model download setup`

## Large Files

- Do not commit model weights, archives, or local admin files.
- Keep fine-tuned model files outside Git history and distribute them through an external release or model host.
