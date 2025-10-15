# Troubleshooting Guide

Common issues and solutions for The Convergence.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Errors](#configuration-errors)
- [API Call Failures](#api-call-failures)
- [Authentication Problems](#authentication-problems)
- [Performance Issues](#performance-issues)
- [Storage & Database](#storage--database)
- [Examples Not Working](#examples-not-working)
- [Import Errors](#import-errors)
- [Getting Help](#getting-help)

---

## Installation Issues

### "Command not found: convergence"

**Problem:** After installation, `convergence` command not found.

**Solution:**

```bash
# Reinstall in development mode
cd /path/to/the-convergence
pip install -e .

# Or check if it's in your PATH
which convergence

# If not in PATH, find the script
find $(python -m site --user-base) -name "convergence"

# Add to PATH if needed
export PATH="$PATH:$(python -m site --user-base)/bin"
```

### "Module not found" errors

**Problem:** Import errors when running commands.

**Solution:**

```bash
# Install all dependencies
pip install -e .

# Or install specific optional features
pip install -e ".[agents]"  # For RLP/SAO
pip install -e ".[dev]"     # For development

# Check installed packages
pip list | grep convergence
```

### Python version compatibility

**Problem:** "Requires Python 3.11 or higher" error.

**Solution:**

```bash
# Check your Python version
python --version

# If too old, install Python 3.11+
# On macOS with Homebrew:
brew install python@3.11

# On Ubuntu:
sudo apt install python3.11

# Create venv with specific version
python3.11 -m venv venv
source venv/bin/activate
pip install the-convergence
```

---

## Configuration Errors

### YAML syntax errors

**Problem:** "Error parsing YAML file" or "mapping values are not allowed here".

**Solution:**

```yaml
# ❌ Bad - Incorrect indentation
api:
name: "test"  # Wrong indent

# ✅ Good - Correct indentation
api:
  name: "test"  # Correct

# ❌ Bad - Missing quotes
search_space:
  parameters:
    model: gpt-4  # Unquoted string with hyphen

# ✅ Good - Quoted when needed
search_space:
  parameters:
    model: "gpt-4"  # Quoted
```

**Validation:**

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Missing required fields

**Problem:** "Field required" or "validation error" messages.

**Solution:**

Every config must have these minimum fields:

```yaml
api:
  name: "my_api"
  endpoint: "https://api.example.com"

search_space:
  parameters: {}  # At least one parameter

evaluation:
  metrics: {}  # At least one metric
```

### Invalid parameter types

**Problem:** "Input should be a valid dictionary" or type errors.

**Solution:**

```yaml
# ❌ Bad - Wrong type
search_space:
  parameters:
    temperature: "0.7"  # String instead of number

# ✅ Good - Correct type
search_space:
  parameters:
    temperature: 0.7  # Number

# For discrete values, use list
search_space:
  parameters:
    model:
      type: "categorical"
      values: ["gpt-4", "gpt-3.5-turbo"]  # List of strings
```

---

## API Call Failures

### "Connection refused" or "Connection timeout"

**Problem:** Cannot connect to API endpoint.

**Solution:**

```bash
# Test endpoint directly
curl -v https://api.openai.com/v1/chat/completions

# Check for proxy issues
export HTTP_PROXY=""
export HTTPS_PROXY=""

# Verify SSL certificates
curl --insecure https://api.openai.com  # If SSL fails

# Check firewall settings
# - Allow outbound HTTPS (port 443)
# - Check corporate proxy settings
```

**In config:**

```yaml
api:
  request:
    timeout_seconds: 60  # Increase timeout
    headers:
      User-Agent: "convergence/0.1.0"  # Some APIs require this
```

### "Rate limit exceeded"

**Problem:** API returns 429 status code.

**Solution:**

```yaml
# Reduce parallelism
optimization:
  execution:
    parallel_workers: 1  # Sequential execution
    experiments_per_generation: 10  # Fewer experiments

# Add delays between calls (future feature)
# For now, use lower worker count
```

### "Invalid response format"

**Problem:** API returns unexpected response structure.

**Solution:**

```yaml
# Configure response parsing
api:
  response:
    success_field: "status"  # Where to find success indicator
    result_field: "data"     # Where to find result
    error_field: "error"     # Where to find errors

# Example response: {"status": "ok", "data": {...}}
```

---

## Authentication Problems

### "Authentication failed" or "Invalid API key"

**Problem:** API key not being sent or is invalid.

**Solution:**

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Verify it's set
echo $OPENAI_API_KEY

# Test with curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Common mistakes:**

```yaml
# ❌ Bad - Hardcoded key (security risk!)
api:
  auth:
    type: "bearer"
    token_env: "sk-hardcoded-key"  # WRONG

# ✅ Good - Environment variable
api:
  auth:
    type: "bearer"
    token_env: "OPENAI_API_KEY"  # Name of env var
```

### "Authorization header missing"

**Problem:** API expects specific header format.

**Solution:**

```yaml
# Bearer token (most common)
api:
  auth:
    type: "bearer"
    token_env: "OPENAI_API_KEY"
    # Sends: Authorization: Bearer sk-...

# API Key in custom header
api:
  auth:
    type: "api_key"
    header_name: "x-api-key"  # Custom header name
    token_env: "MY_API_KEY"
    # Sends: x-api-key: your-key

# Basic auth
api:
  auth:
    type: "basic"
    username: "user"
    password_env: "API_PASSWORD"
    # Sends: Authorization: Basic base64(user:pass)
```

### Environment variables not loading

**Problem:** .env file exists but variables not loaded.

**Solution:**

```bash
# Install python-dotenv
pip install python-dotenv

# Verify .env format (no spaces around =)
# ❌ Bad
OPENAI_API_KEY = "sk-..."  # Spaces around =

# ✅ Good
OPENAI_API_KEY="sk-..."  # No spaces

# Load manually if needed
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

---

## Performance Issues

### Optimization runs too slow

**Problem:** Optimization takes hours to complete.

**Solution:**

```yaml
# Enable parallel execution
optimization:
  execution:
    parallel_workers: 5  # Run 5 tests in parallel
    experiments_per_generation: 20  # Fewer experiments

# Reduce generations
optimization:
  evolution:
    generations: 5  # Down from 10
    population_size: 10  # Down from 20

# Enable early stopping
optimization:
  execution:
    early_stopping:
      enabled: true
      patience: 2  # Stop after 2 generations without improvement
```

### Memory usage too high

**Problem:** Python process consuming too much RAM.

**Solution:**

```yaml
# Use file storage instead of memory
society:
  storage:
    backend: "file"  # Not "memory"
    path: "./data/optimization"

# Disable caching
society:
  storage:
    cache_enabled: false

# Reduce result retention
output:
  save_all_experiments: false  # Only save best config
```

### Disk space filling up

**Problem:** Database/results consuming too much disk.

**Solution:**

```bash
# Check disk usage
du -sh data/ results/ legacy/

# Clean old results
rm -rf results/*/  # Remove all result folders

# Vacuum database
sqlite3 data/legacy.db "VACUUM;"

# Or start fresh
rm -rf data/ results/ legacy/
```

---

## Storage & Database

### "Database is locked"

**Problem:** SQLite database locked by another process.

**Solution:**

```bash
# Find processes using the database
lsof data/legacy.db

# Kill stale processes
pkill -f convergence

# Or use different database
convergence optimize config.yaml  # Will create new session
```

### "Table already exists" error

**Problem:** Database schema mismatch.

**Solution:**

```bash
# Backup current database
cp data/legacy.db data/legacy.db.backup

# Reset database (loses history!)
rm data/legacy.db

# Or migrate (future feature)
# convergence migrate --from 0.0.1 --to 0.1.0
```

### Results not being saved

**Problem:** No results in output directory.

**Solution:**

```yaml
# Check output configuration
output:
  save_path: "./results/optimization_run"  # Must be writable
  save_all_experiments: true
  formats: ["json", "markdown", "csv"]  # Enable all formats

# Verify directory permissions
```

```bash
chmod 755 results/
mkdir -p results/optimization_run
```

---

## Examples Not Working

### Example fails with "File not found"

**Problem:** Test cases or evaluator files missing.

**Solution:**

```bash
# Run from correct directory
cd examples/ai/openai/
convergence optimize openai_responses_optimization.yaml

# Or use absolute paths in YAML
evaluation:
  test_cases:
    path: "/full/path/to/tests.json"
  custom_evaluator:
    module: "/full/path/to/evaluator.py"
```

### "API key not set" in examples

**Problem:** Example requires API key but not set.

**Solution:**

```bash
# Check example's README for required keys
cat examples/ai/openai/OPENAI_SETUP.md

# Set required keys
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export BROWSERBASE_API_KEY="..."

# Or create .env file in project root
echo "OPENAI_API_KEY=sk-..." > .env
```

### Example runs but all tests fail

**Problem:** API endpoint changed or tests are outdated.

**Solution:**

```bash
# Test API directly first
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'

# If API works, update test cases in tests.json

# If API format changed, update the YAML config
```

---

## Import Errors

### "Cannot import name 'OptimizationRunner'"

**Problem:** Module import failing.

**Solution:**

```bash
# Reinstall package
pip uninstall the-convergence
pip install -e .

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### "No module named 'weave'"

**Problem:** Optional dependency not installed.

**Solution:**

```bash
# Weave is required but might fail to install
pip install weave

# If Weave installation fails, it's optional
# System will continue without it (observability disabled)

# To force it:
pip install weave --upgrade
```

### "No module named 'transformers'"

**Problem:** Agent society dependencies not installed.

**Solution:**

```bash
# Install optional agent features
pip install "the-convergence[agents]"

# Or install individually
pip install transformers torch trl

# Agent society features are optional
# Disable in config if not needed:
```

```yaml
society:
  enabled: false  # Disable agent features
```

---

## Getting Help

### Check logs

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
convergence optimize config.yaml --verbose

# Check log files
tail -f logs/convergence.log  # If logging to file
```

### Minimal reproduction

When reporting issues, provide:

```bash
# System info
python --version
pip show the-convergence
uname -a  # Or: ver (Windows)

# Minimal config that reproduces issue
cat minimal_config.yaml

# Full error message
convergence optimize minimal_config.yaml 2>&1 | tee error.log
```

### Common debugging commands

```bash
# Validate configuration
python -c "from convergence.optimization.config_loader import ConfigLoader; ConfigLoader.load('config.yaml')"

# Test API connection
python -c "import httpx; print(httpx.get('https://api.openai.com/v1/models').status_code)"

# Check environment variables
env | grep -i api

# Test imports
python -c "from convergence.optimization.runner import OptimizationRunner; print('OK')"
```

### Still stuck?

1. **Search existing issues:** [GitHub Issues](https://github.com/persist-os/the-convergence/issues)
2. **Ask in discussions:** [GitHub Discussions](https://github.com/persist-os/the-convergence/discussions)
3. **Open new issue:** Provide:
   - Python version
   - Convergence version
   - Full error message
   - Minimal config to reproduce
   - What you've tried

---

## FAQ

### Q: Can I use this without API keys?

**A:** Yes! Use `mock_mode`:

```yaml
api:
  mock_mode: true  # Skip real API calls
```

### Q: How do I optimize for one metric only?

**A:** Set other weights to 0:

```yaml
evaluation:
  metrics:
    quality: {weight: 1.0}
    cost: {weight: 0.0}     # Ignored
    latency: {weight: 0.0}  # Ignored
```

### Q: Can I resume optimization if it crashes?

**A:** Yes, if legacy tracking is enabled:

```yaml
legacy:
  enabled: true
  session_id: "my_session"  # Use same ID to resume
```

### Q: How do I test changes without wasting API credits?

**A:** Use mock mode or small populations:

```yaml
optimization:
  evolution:
    generations: 1
    population_size: 2
api:
  mock_mode: true  # No real API calls
```

### Q: Can I use this with non-OpenAI APIs?

**A:** Yes! Any REST API works:

```yaml
api:
  name: "custom_api"
  endpoint: "https://your-api.com/endpoint"
  # Configure authentication and response format
```

---

**Last Updated:** October 15, 2025  
**Version:** 1.0  

For more help, see:
- [README.md](README.md) - Main documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [examples/](examples/) - Working examples

