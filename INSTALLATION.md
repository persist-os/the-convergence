# Convergence Installation Guide

## Package Names

- **Package name** (for pip): `the-convergence`
- **Import name** (in Python): `convergence`

```bash
# Install
pip install the-convergence

# Import
from convergence import run_optimization
```

## Installation Methods

### 1. Development Install (Editable)

For local development where you want changes to reflect immediately:

```bash
cd /path/to/the-convergence
pip install -e .
```

This creates a symbolic link, so code changes take effect immediately without reinstalling.

### 2. Backend Integration

For the backend to use Convergence:

```bash
# Already in backend/requirements.txt as:
the-convergence

# Install dependencies
cd /path/to/backend
pip install -r requirements.txt
```

### 3. Production Install

```bash
pip install the-convergence
```

## Verify Installation

```bash
# Test 1: Import check
python -c "from convergence import run_optimization; print('✅ SDK import works')"

# Test 2: Check version
python -c "from convergence import __version__; print(f'Convergence v{__version__}')"

# Test 3: CLI check
convergence --help
```

## Installing from Source

If you need the latest development version:

```bash
# Clone repository
git clone https://github.com/persist-os/the-convergence.git
cd the-convergence

# Install in development mode
pip install -e .

# Or build and install
pip install build
python -m build
pip install dist/the_convergence-*.whl
```

## Backend Integration Setup

### Step 1: Install Package

```bash
cd backend
pip install -e ../the-convergence
```

### Step 2: Verify Import

```python
# Test in Python
from convergence import run_optimization
print("✅ Convergence SDK ready")
```

### Step 3: Update Requirements (if needed)

The backend's `requirements.txt` already includes:
```
the-convergence
```

If you need a specific version:
```
the-convergence>=0.1.2
```

## Common Issues

### Issue: Import Error

```python
ImportError: No module named 'convergence'
```

**Solution**: Package not installed. Run:
```bash
pip install the-convergence
```

### Issue: Wrong Python Environment

```bash
# Check which Python/pip you're using
which python
which pip

# Use correct environment
source venv/bin/activate  # if using venv
pip install the-convergence
```

### Issue: Editable Install Not Working

```bash
# Reinstall in editable mode
pip uninstall the-convergence
cd /path/to/the-convergence
pip install -e .
```

## Development Workflow

### For Convergence Development

```bash
cd the-convergence
pip install -e .
# Make changes to code
# Test immediately (no reinstall needed)
python test_script.py
```

### For Backend Development

```bash
# Install Convergence in editable mode
cd backend
pip install -e ../the-convergence

# Now backend can import latest Convergence code
python -m app.background_jobs.executors.convergence_optimization
```

## Dependencies

Convergence requires:
- Python >= 3.11
- pydantic >= 2.0.0
- httpx >= 0.25.0
- pyyaml >= 6.0.0
- Other dependencies (see pyproject.toml)

These are automatically installed when you install `the-convergence`.

## Next Steps

After installation, see:
- `SDK_USAGE.md` - How to use the SDK
- `examples/` - Working examples
- `README.md` - Full documentation

