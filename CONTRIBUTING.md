# Contributing Guidelines

Thank you for your interest in contributing to PriorAnchor. We welcome pull requests, bug reports, and architectural proposals.

## Code Standards and Style

- **Python**: Python 3.10+ compatibility. Code must be formatted and linted with `ruff` (`ruff check src tests examples`).
- **Typing**: Strict type annotations are enforced using standard Python typing primitives.
- **Testing**: All contributions must include unit tests under `tests/`. New features must achieve 100% test pass rate (`pytest -v`).
- **Zero-Emoji Policy**: Zero emojis are permitted in code, commit messages, documentation, or pull requests.

## Development Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Set up the development virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run tests and static analysis:
   ```bash
   ruff check src tests
   pytest -v
   ```
4. Commit your changes with descriptive Conventional Commits messages referencing relevant issue numbers:
   ```bash
   git commit -m "feat(core): implement adaptive loss weighting (closes #123)"
   ```
5. Submit a pull request to `main`.
