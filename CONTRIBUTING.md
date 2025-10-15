# Contributing to The Convergence

Thank you for your interest in contributing to The Convergence! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone. We expect all contributors to:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or trolling
- Personal attacks or inflammatory comments
- Spam or self-promotion
- Publishing private information without consent

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of async Python
- Familiarity with API optimization concepts (helpful but not required)

### Development Setup

1. **Fork and Clone**

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/the-convergence.git
cd the-convergence
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in Development Mode**

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

4. **Set Up Pre-commit Hooks** (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

5. **Verify Installation**

```bash
convergence --version
convergence info
```

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

#### 🐛 Bug Reports

- Use GitHub Issues
- Include minimal reproduction steps
- Specify your environment (OS, Python version, etc.)
- Attach relevant error messages and logs

#### ✨ Feature Requests

- Open a GitHub Issue with [Feature Request] tag
- Describe the use case
- Explain why it would benefit users
- Consider starting a discussion before implementing

#### 📖 Documentation

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation (future)

#### 💻 Code Contributions

- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

#### 🎨 Examples

- Add optimization examples for new APIs
- Improve existing examples
- Add custom evaluators
- Create tutorials

### Finding Issues to Work On

- Look for issues tagged `good first issue` or `help wanted`
- Check the project board for planned features
- Ask in discussions if you're unsure where to start

## Coding Standards

### Python Style

We follow **PEP 8** with these conventions:

- **Line Length:** 100 characters (not 80)
- **Quotes:** Double quotes for strings, single for dict keys
- **Type Hints:** Required for all public functions
- **Async:** Use async/await consistently

### Code Organization

```python
# Standard library imports
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
from pydantic import BaseModel
import weave

# Local imports
from convergence.core.protocols import LLMProvider
from convergence.optimization.models import OptimizationResult
```

### Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions:** `snake_case()`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private:** `_leading_underscore`

### Documentation Style

```python
def optimize_api(
    config: OptimizationSchema,
    test_cases: List[Dict[str, Any]]
) -> OptimizationResult:
    """
    Optimize API configuration using evolutionary algorithms.
    
    This function runs the complete optimization pipeline including:
    - MAB-based exploration
    - Genetic algorithm evolution
    - Multi-objective evaluation
    
    Args:
        config: Complete optimization configuration
        test_cases: List of test cases to evaluate against
        
    Returns:
        OptimizationResult with best config and metrics
        
    Raises:
        ConfigurationError: If config is invalid
        APIError: If API calls fail consistently
        
    Example:
        >>> config = ConfigLoader.load("optimization.yaml")
        >>> result = await optimize_api(config, test_cases)
        >>> print(result.best_config)
    """
    # Implementation
```

### Type Hints

Always use type hints for public APIs:

```python
# ✅ Good
async def fetch_data(url: str, timeout: int = 30) -> Dict[str, Any]:
    ...

# ❌ Bad
async def fetch_data(url, timeout=30):
    ...
```

### Error Handling

```python
# ✅ Good - Specific exceptions with context
try:
    result = await api_caller.call(endpoint, params)
except APIError as e:
    logger.error(f"API call failed: {e}")
    raise OptimizationError(f"Could not complete optimization: {e}") from e

# ❌ Bad - Bare except
try:
    result = api_caller.call(endpoint, params)
except:
    return None
```

### Async Best Practices

```python
# ✅ Good - Properly awaited
async def process_batch(items: List[str]) -> List[Result]:
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)

# ❌ Bad - Not awaited
async def process_batch(items: List[str]) -> List[Result]:
    return [process_item(item) for item in items]  # Returns coroutines!
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=convergence --cov-report=html

# Run specific test file
pytest tests/test_optimization.py

# Run with verbose output
pytest -v
```

### Writing Tests

```python
import pytest
from convergence.optimization.runner import OptimizationRunner

@pytest.mark.asyncio
async def test_optimization_basic():
    """Test basic optimization workflow."""
    # Arrange
    config = create_test_config()
    runner = OptimizationRunner(config)
    
    # Act
    result = await runner.run()
    
    # Assert
    assert result.best_score > 0
    assert len(result.all_results) > 0
    assert result.best_config is not None

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return OptimizationSchema(
        api=APIConfig(name="test", endpoint="https://api.test.com"),
        search_space=SearchSpaceConfig(parameters={}),
        evaluation=EvaluationConfig(metrics={})
    )
```

### Test Coverage

- Aim for >80% coverage on new code
- Focus on edge cases and error paths
- Include integration tests for critical flows
- Mock external API calls

## Documentation

### README Updates

- Update README.md for user-facing features
- Keep examples up to date
- Update table of contents
- Check all links work

### Docstrings

- Required for all public functions, classes, and modules
- Use Google-style docstrings
- Include examples for complex functions
- Document exceptions that can be raised

### Examples

When adding examples:

1. **Create example directory:**

   ```
   examples/my_api/
   ├── my_api_optimization.yaml
   ├── my_api_evaluator.py
   ├── my_api_tests.json
   └── README.md
   ```

2. **Document API setup:**
   - How to get API keys
   - Any prerequisites
   - Expected costs

3. **Test your example:**
   - Run the optimization
   - Verify results make sense
   - Include sample output

4. **Add to main examples README**

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] Type hints added
- [ ] No merge conflicts

### PR Title Format

Use conventional commits:

- `feat: Add support for Anthropic Claude`
- `fix: Handle timeout errors in API caller`
- `docs: Update contributing guidelines`
- `test: Add tests for evolution engine`
- `refactor: Simplify legacy store queries`
- `perf: Optimize parallel execution`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of changes
- Made in this PR

## Testing
How was this tested?

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Follows style guide
```

### Review Process

1. **Submit PR** - GitHub will automatically run CI checks
2. **Address feedback** - Maintainers will review within 2-3 days
3. **Make changes** - Push updates to your branch
4. **Approval** - At least one maintainer approval required
5. **Merge** - Maintainers will merge when ready

### After Merge

- Your contribution will be included in the next release
- You'll be added to CONTRIBUTORS.md
- Thank you! 🎉

## Community

### Getting Help

- **GitHub Discussions:** Ask questions, share ideas
- **GitHub Issues:** Bug reports and feature requests
- **Examples:** Check `examples/` directory for reference
- **Documentation:** Read docs in `documentation/` directory

### Stay Updated

- Watch the repository for notifications
- Read CHANGELOG.md for updates
- Follow release notes

### Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page

## Development Tips

### Useful Commands

```bash
# Format code
black convergence/

# Check types
mypy convergence/

# Lint
ruff check convergence/

# Run single example
convergence optimize examples/ai/openai/openai_responses_optimization.yaml

# Build package
python -m build

# Install local build
pip install dist/the_convergence-*.whl
```

### Debugging

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
convergence optimize config.yaml --verbose

# Use Python debugger
import pdb; pdb.set_trace()

# Check Weave traces
# Visit: https://wandb.ai/your-org/convergence
```

## Questions?

If you have questions not covered here:

1. Check existing issues and discussions
2. Read the documentation in `documentation/`
3. Ask in GitHub Discussions
4. Open an issue with the `question` label

## Thank You

Every contribution matters, whether it's:

- Reporting a bug
- Fixing a typo
- Adding a feature
- Improving documentation

Your work makes The Convergence better for everyone.

**Happy contributing!** 🚀

---

*Last updated: October 15, 2025*
