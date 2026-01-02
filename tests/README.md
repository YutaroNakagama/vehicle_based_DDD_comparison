# Tests

This directory contains the test code for the project.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run a specific test file
```bash
pytest tests/test_config.py
```

### Run a specific test function
```bash
pytest tests/test_config.py::test_paths_exist
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Filter by markers
```bash
# Fast tests only
pytest tests/ -m fast

# Smoke tests only
pytest tests/ -m smoke

# Exclude slow tests
pytest tests/ -m "not slow"
```

## Test Structure

- `conftest.py`: Common pytest fixtures (sample data generation, mock environment setup, etc.)
- `test_config.py`: Configuration file and path tests
- `test_imbalance_methods.py`: Imbalanced data handling method tests (SMOTE, undersampling, etc.)
- `test_smoke_pipeline.py`: Lightweight end-to-end tests for data processing
- `test_smoke_training.py`: Lightweight end-to-end tests for model training
- `fixtures/`: Test dummy data

## Test Marker Descriptions

- `@pytest.mark.fast`: Fast tests that complete within seconds (default)
- `@pytest.mark.slow`: Tests that may take more than 1 minute
- `@pytest.mark.smoke`: Lightweight tests to verify the entire pipeline works
- `@pytest.mark.integration`: Tests that integrate multiple components
- `@pytest.mark.requires_data`: Tests that require actual datasets

## Test Data

Dummy data for testing is placed in the `tests/fixtures/` directory.
Tests using real data can be run by specifying the data path via environment variables.
