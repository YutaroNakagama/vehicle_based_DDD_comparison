# Documentation

This directory contains Sphinx documentation for the Vehicle-Based Driver Drowsiness Detection project.

## Structure

```
docs/
├── conf.py              # Sphinx configuration
├── Makefile            # Build script (Unix)
├── make.bat            # Build script (Windows)
├── requirements.txt    # Documentation dependencies
└── source/             # Documentation source files
    ├── index.rst       # Main documentation entry point
    ├── api/            # API reference documentation
    │   ├── modules.rst # Module index
    │   └── utils.rst   # Utils package structure
    └── guide/          # User guides
        ├── developer_guide.md              # Developer guide and architecture
        ├── data_pipeline.rst               # Data preprocessing pipeline
        ├── models.rst                      # Model architectures
        ├── evaluation.rst                  # Evaluation framework
        ├── analysis.rst                    # Analysis tools
        └── domain_generalization_pipeline.md # Domain generalization workflow
```

## Building Documentation

### Install Dependencies

```bash
cd docs
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
make html
```

The generated documentation will be available at `docs/_build/html/index.html`.

## Notes

- API documentation is auto-generated from docstrings using Sphinx autodoc
- Markdown files are supported via myst_parser extension
- The documentation uses the Read the Docs theme
