# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python package with submodules for preprocessing, training, evaluation, deployment, monitoring, and the SageMaker pipeline.
- `scripts/` contains CLI utilities for data download, S3 upload, and pipeline deployment.
- `tests/` includes pytest-based unit tests (e.g., `tests/test_training.py`).
- `config/` stores YAML configuration for models and pipelines.
- `infrastructure/terraform/` and `infrastructure/cloudformation/` provide AWS IaC.
- `data/` and `models/` are runtime artifacts; expect local or S3-backed content.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt && pip install -e .`: install runtime and dev dependencies.
- `python scripts/download_data.py --dataset isic2019 --output-dir data/raw`: fetch datasets.
- `python scripts/upload_to_s3.py --bucket <bucket>`: push data to S3 for training.
- `python scripts/deploy_pipeline.py --config config/config.yaml`: deploy the SageMaker pipeline.

## Coding Style & Naming Conventions
- Language: Python 3.8+.
- Formatters/linters: Black, isort, Flake8, pylint, and mypy are listed in `requirements.txt`.
- Use standard Python naming: `snake_case` for functions/vars, `PascalCase` for classes, and `test_*.py` for tests.
- Keep modules focused; prefer adding new steps under `src/pipeline/` rather than bloating scripts.

## Testing Guidelines
- Framework: pytest (with pytest-cov, pytest-mock available).
- Run tests with `pytest` or `python -m pytest -v`.
- Name tests after the behavior, e.g., `test_preprocessing_handles_empty_input`.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and descriptive (e.g., "Add quick start guide").
- PRs should include a concise summary, testing notes, and any relevant screenshots/diagrams for docs changes.

## Security & Configuration Tips
- Configure AWS access with `aws configure` before running pipeline or infra scripts.
- Keep sensitive data out of git; use `data/`, `models/`, and S3 for large artifacts.
- Update `config/*.yaml` rather than hardcoding AWS or model settings.
