# raw2ultra

A Python CLI tool for raw2ultra processing.

## Installation

### Using Poetry (Development)

```bash
# Clone the repository
git clone <repository-url>
cd raw2ultra

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip (Production)

```bash
pip install raw2ultra
```

## Usage

### Basic Commands

```bash
# Show help
raw2ultra --help

# Show version
raw2ultra --version

# Show information about the tool
raw2ultra info

# Process a file (basic usage)
raw2ultra process input_file.raw

# Process a file with output specification
raw2ultra process input_file.raw --output output_file.ultra

# Enable verbose output
raw2ultra --verbose process input_file.raw
```

### Available Commands

- `process`: Process a raw file to ultra format
- `info`: Show information about raw2ultra

### Options

- `--verbose, -v`: Enable verbose output
- `--help`: Show help message
- `--version`: Show version information

## Development

### Running Tests

```bash
poetry run pytest
```

### Project Structure

```
raw2ultra/
├── raw2ultra/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── cli.py          # CLI interface
│   └── main.py         # Main entry point
├── tests/              # Test files
│   ├── __init__.py
│   └── test_cli.py
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Requirements

- Python 3.10+
- click >= 8.2.1

## License

See LICENSE file for details.
