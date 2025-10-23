# Repository Guidelines

## Project Structure & Module Organization
Code lives in `snn_ocr/`, with modules such as `bitfont.py`, `synth.py`, `spikes.py`, `lif.py`, `otc.py`, `seq.py`, and `ctc.py` reflecting the pipeline from rendering to decoding. Generated curricula stay under `data/` as stage-prefixed folders (`s1_digits`, `s2_letters`, `s3_words`, `s4_sentences`); checkpoints and logs mirror that naming in `runs/`. Keep exploratory notebooks or profiling traces in `experiments/`, and share reusable helpers through `snn_ocr/utils.py`. Route new commands through `cli.py` so contributors keep the single `python -m snn_ocr.cli` entry point.

## Build, Test, and Development Commands
Only the standard library is allowed. Typical operations:
- `python -m snn_ocr.cli synth --stage S1 --n 2000 --out data/s1_digits` builds curriculum data.
- `python -m snn_ocr.cli train --stage S3 --data data/s3_words --epochs 10 --beam 5` trains stage modules.
- `python -m snn_ocr.cli eval --stage S4 --data data/s4_sentences --ckpt runs/s4/ckpt.json` reports CER/WER.
- `python -m unittest discover -s tests -p "test_*.py"` runs automated checks (add `-v` when debugging).

## Coding Style & Naming Conventions
Stick to PEP 8, 4-space indentation, and descriptive type hints. Name classes after their role (`LIFBlock`, `OpticalTokenCompressor`), keep filenames snake_case, and document stage-specific behavior with brief docstrings. If you need performance helpers, implement them in pure Python and highlight any hot loops with short comments.

## Testing Guidelines
Mirror module names inside `tests/` (for example `test_lif.py`, `test_ctc.py`). New stages or encoders should ship deterministic fixtures (use fixed seeds) plus assertions on spike counts, tensor shapes, and loss trends. Run `python -m unittest` before every PR and note runtimes when regressions are likely. Aim for at least smoke coverage across S1–S4 flows whenever you touch shared infrastructure.

## Commit & Pull Request Guidelines
There is no established history yet; prefer imperative messages formatted `stage-scope: action` (e.g. `s3-training: add adaptive otc pooling`) and reference issue IDs when available. PRs should include the intent, CLI commands executed (`synth`, `train`, `eval`), resulting metrics, and artifacts such as preview PGM paths. Seek review for architectural changes and wait for a green `python -m unittest` run before merging.

## Security & Configuration Tips
Stay within the standard library unless the maintainer approves a dependency. Default new CLI outputs to subfolders of `data/` or `runs/`, guard against overwriting existing runs, and scrub generated text samples before committing datasets.
