# Contributing

Thank you for your interest in contributing to NWM Coastal! This guide will help you get
started.

## Development Setup

### Prerequisites

- Python >= 3.11
- [Pixi](https://pixi.prefix.dev/latest/) for environment management
- Git

!!! tip "Installing Pixi"
    If you don't have Pixi installed, see the
    [Installation Guide](getting-started/installation.md#development-installation-with-pixi)
    for setup instructions. Remember to restart your terminal after installing Pixi.

### Clone and Install

```bash
git clone https://github.com/NGWPC/nwm-coastal
cd nwm-coastal
pixi install -e dev
```

### Verify Installation

```bash
pixi r -e dev coastal-calibration --help
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pixi r -e test314 test

# Run with coverage report
pixi r -e test314 report
```

### Type Checking

```bash
pixi r -e typecheck typecheck
```

### Linting

```bash
pixi r lint
```

This runs pre-commit hooks including:

- Ruff (linting and formatting)
- Codespell (spelling)
- YAML/JSON formatting
- Various file checks

### Building Documentation

```bash
pixi r -e docs docs-serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- Type hints are required for all public functions
- Maximum line length: 100 characters

### Docstring Example

```python
def validate_date_ranges(
    start_time: datetime,
    end_time: datetime,
    meteo_source: str,
    boundary_source: str,
    coastal_domain: str,
) -> list[str]:
    """Validate simulation dates against available data source ranges.

    Parameters
    ----------
    start_time : datetime
        Simulation start time.
    end_time : datetime
        Simulation end time.
    meteo_source : str
        Meteorological data source ('nwm_ana' or 'nwm_retro').
    boundary_source : str
        Boundary condition source ('tpxo' or 'stofs').
    coastal_domain : str
        Coastal domain identifier.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.

    Examples
    --------
    >>> errors = validate_date_ranges(
    ...     datetime(2021, 6, 11),
    ...     datetime(2021, 6, 12),
    ...     "nwm_ana",
    ...     "stofs",
    ...     "hawaii"
    ... )
    >>> len(errors)
    0
    """
```

## Pull Request Process

1. **Fork** the repository
1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
1. **Make your changes** with tests
1. **Run all checks**:
   ```bash
   pixi r lint
   pixi r -e typecheck typecheck
   pixi r -e test314 test
   ```
1. **Commit** with a descriptive message:
   ```bash
   git commit -m "Add feature X that does Y"
   ```
1. **Push** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
1. **Open a Pull Request** against the `main` branch

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues when applicable ("Fix #123")

### PR Checklist

- [ ] Tests pass locally
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (for user-facing changes)

## Reporting Issues

### Bug Reports

Include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:

- Use case description
- Proposed solution (if any)
- Alternatives considered

## Project Structure

```
nwm-coastal/
├── src/coastal_calibration/    # Main package
│   ├── config/                 # Configuration handling
│   ├── stages/                 # Workflow stages
│   ├── utils/                  # Utilities
│   ├── scripts/                # Legacy scripts (excluded from linting)
│   ├── cli.py                  # CLI entry point
│   ├── runner.py               # Workflow runner
│   └── downloader.py           # Data download
├── docs/                       # Documentation
├── examples/                   # Example configurations
├── tests/                      # Test suite
├── pyproject.toml              # Project configuration
└── mkdocs.yml                  # Documentation configuration
```

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion

## License

By contributing, you agree that your contributions will be licensed under the
BSD-2-Clause License.
