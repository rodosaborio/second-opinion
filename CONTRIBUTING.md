# Contributing to Second Opinion

Thank you for your interest in contributing to Second Opinion! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be professional and considerate in all interactions.

## License Agreement

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0. This includes an explicit patent grant as outlined in the license terms.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/second-opinion.git
   cd second-opinion
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/rodosaborio/second-opinion.git
   ```

## Development Setup

1. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

3. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys for testing
   ```

4. **Run setup validation**:
   ```bash
   uv run python setup_guide.py
   ```

## Making Changes

### Branch Strategy

- Create feature branches from `main`
- Use descriptive branch names: `feature/add-new-tool`, `fix/rate-limiting-bug`, `docs/improve-readme`

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

### Development Workflow

1. **Make your changes**
2. **Add/update tests** for new functionality
3. **Update documentation** if needed
4. **Run tests and linting** (see below)
5. **Commit your changes** with clear messages

## Testing

### Running Tests

```bash
# Fast development testing (recommended during development)
uv run pytest

# Full coverage testing
uv run pytest --cov=second_opinion --cov-report=html --cov-fail-under=85

# Run specific test categories
uv run pytest -m security     # Security tests
uv run pytest -m integration  # Integration tests
uv run pytest -m "not slow"   # Skip slow tests

# Debug failing tests
uv run pytest -v --tb=long
```

### Test Requirements

- **All tests must pass** before submitting a PR
- **Maintain 85%+ code coverage** for new code
- **Add tests for new features** and bug fixes
- **Use appropriate test markers** (`security`, `integration`, `slow`)

### Writing Tests

- Follow existing test patterns in the `tests/` directory
- Use fixtures for common setup (see `conftest.py`)
- Mock external dependencies (API calls, file system, etc.)
- Write both unit and integration tests where appropriate

## Code Style

We use automated code formatting and linting:

### Automated Formatting (Pre-commit)

Pre-commit hooks automatically format your code:
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **MyPy** for type checking

### Manual Code Quality Checks

```bash
# Format code
uv run black .

# Lint code
uv run ruff check . --fix

# Type checking
uv run mypy src/

# Run all quality checks
uv run black . && uv run ruff check . --fix && uv run mypy src/
```

### Code Style Guidelines

- **Use type hints** for all function parameters and return values
- **Write docstrings** for public functions and classes
- **Follow PEP 8** conventions (enforced by Black)
- **Keep functions focused** and small when possible
- **Use meaningful variable names**
- **Add comments** for complex logic, but prefer self-documenting code

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass** and code is properly formatted
2. **Update documentation** if you've changed functionality
3. **Update CHANGELOG.md** with your changes
4. **Commit your changes** with clear, descriptive messages
5. **Push to your fork** and create a pull request

### Pull Request Guidelines

- **Use a clear PR title** that describes the change
- **Fill out the PR template** completely
- **Link to related issues** if applicable
- **Request review** from maintainers
- **Be responsive** to feedback and requested changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add model benchmark comparison tool
fix: resolve rate limiting issues with OpenRouter
docs: improve API documentation for MCP tools
test: add integration tests for cost tracking
refactor: simplify client factory pattern
```

### PR Checklist

Before submitting your PR, ensure:

- [ ] All tests pass (`uv run pytest`)
- [ ] Code coverage is maintained (`uv run pytest --cov`)
- [ ] Code is properly formatted (pre-commit handles this)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] PR description clearly explains the changes

## Development Guidelines

### Security Considerations

- **Never commit real API keys** or sensitive data
- **Use environment variables** for configuration
- **Validate all inputs** properly
- **Follow security best practices** in code reviews

### Architecture Guidelines

- **Follow existing patterns** in the codebase
- **Use dependency injection** for testability
- **Implement proper error handling**
- **Add logging** for debugging and monitoring
- **Consider performance implications** of changes

### Adding New Features

1. **Discuss major changes** in an issue first
2. **Follow the MCP tool blueprint** for new tools (see CLAUDE.md)
3. **Add comprehensive tests**
4. **Update documentation**
5. **Consider backward compatibility**

## Release Process

Releases are managed by maintainers, but contributors should:

1. **Update CHANGELOG.md** with their changes
2. **Ensure version compatibility** is maintained
3. **Test thoroughly** before marking PR ready

## Getting Help

- **Check existing issues** for similar problems
- **Review documentation** in README.md and CLAUDE.md
- **Ask questions** in issue discussions
- **Be specific** about your environment and steps to reproduce issues

## Recognition

Contributors are recognized in:
- **CHANGELOG.md** for significant contributions
- **GitHub contributors page**
- **Release notes** for major features

---

Thank you for contributing to Second Opinion! Your efforts help make AI tools more accessible and cost-effective for everyone.