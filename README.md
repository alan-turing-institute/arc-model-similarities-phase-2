# ARC Model Similarity Phase 2

## Problem Statement

Previous ARC research examined the relationship between neural network similarity and attack transferability between those two networks.

In the second phase of model similarity research, we are now assessing the relationship between dataset similarity and attack transferability between identitical models trained on two different datasets.

We expect that dataset similarity will be predictive of attack transferability. This is in line with recent research that shows dataset similarity is predictive of successful transfer learning.

**Statement of Work (internal only):** <https://thealanturininstitute.sharepoint.com/:b:/s/tds/Ee5knRguzUxChT-aziaBP3QBAQppT8YrDpYv-TsSrkRn1g?e=jei47a>

## Technical Approach

- **Step 1:** Generate target-surrogate pairs of datasets with differences between them
  - Start with smaller-scale differences
- **Step 2:** Compute similarity metrics between target-surrogate pairs
- **Step 3:** Train a network holding architecture and hyperparameters constant on each pair
  - We may relax the requirement that tuning be identical in the future
- **Step 4:** Perform transfer attack. Record transfer attack success metrics
- **Step 5:** Assess relationship between similarity metrics and transfer attack succes

## Installation

1. Clone this repository

2. Install with `pip`:

   ```bash
   pip install .
   ```

## Usage

You can begin using the package code with an import command:

   ```python
   import modsim2
   ```

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

### Testing

1. To run tests:

    ```bash
    pytest tests
    ```

### Linters

- If you have setup pre-commit `flake8`, `black`, and `isort` will run automatically before making commits
- Or you can run them manually:

    ```bash
    poetry run black .
    poetry run isort .
    poetry run flake8
    ```

### Adding to the Package

- Your source code files should go in the `src/modsim2` directory. These will be available as a python package, i.e. you can do `from modsim2 import myfunction` etc.
- Add tests (in files with names like `test_*.py` and with functions with names starting `test_*`) the `tests/` directory.
