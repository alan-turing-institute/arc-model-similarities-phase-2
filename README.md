# ARC Model Similarity Phase 2

## Installation

1. Clone this repository

2. Install with `pip`:

   ```bash
   pip install .
   ```

## Usage

**TODO**

## Development

### Developer Setup

1. Install dependencies with Poetry

   ```bash
   poetry install
   ```

2. Install pre-commit hooks:

   ```bash
   poetry run pre-commit install --install-hooks
   ```

3. Note that per issue [#8] you may encounter difficulties in installing pytorch with the current setup (i.e. if you are not running a mac with arm architecture). If this is the case, please update the URLs on lines 13 and 14 of pyproject.toml

### Testing

1. To run tests:

    ```bash
    pytest tests
    ```
