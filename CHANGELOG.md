# Changelog

All notable changes to Second Opinion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-06-21

### Added
- Public release preparation
- Apache 2.0 License (provides patent protection)
- Comprehensive documentation (CONTRIBUTING.md, SECURITY.md)
- GitHub Actions CI/CD pipeline
- Issue and PR templates
- Dependabot configuration

## [0.1.0] - 2025-06-21

### Added
- Core second opinion functionality for AI model comparison
- MCP (Model Context Protocol) server implementation
- Multiple MCP tools:
  - `second_opinion`: Core model comparison and recommendation
  - `should_downgrade`: Cost optimization through cheaper alternatives
  - `should_upgrade`: Quality improvement analysis for premium models
  - `compare_responses`: Detailed side-by-side response analysis
  - `consult`: AI consultation and task delegation
- Support for 1,100+ models through OpenRouter
- Comprehensive cost tracking and budget protection
- Security features:
  - Input sanitization to prevent prompt injection
  - Response filtering to prevent API key leakage
  - Rate limiting and concurrent request management
  - Data encryption for local storage
- CLI interface with rich formatting and verbose modes
- Configuration management with YAML and environment variables
- Professional test suite with 85%+ coverage
- Complete development tooling (pre-commit hooks, linting, type checking)

### Features
- **Multi-model support**: Access to OpenAI, Anthropic, Google, Meta, Cohere, and more
- **Cost optimization**: Built-in cost estimation, limits, and recommendations
- **Response reuse**: Save API costs by reusing existing responses
- **Context-aware recommendations**: Model suggestions based on task type
- **Verbose mode**: Full response analysis for complex tasks
- **Local model support**: Integration with LM Studio for local models
- **Session management**: Cost tracking and conversation context
- **Setup validation**: Interactive setup guide with connection testing

### Security
- Input sanitization enabled by default
- Response filtering to prevent sensitive data leakage
- API key validation and secure storage
- Encryption support for local data
- Rate limiting to prevent abuse
- Cost guards to prevent runaway spending

### Documentation
- Comprehensive README with examples and troubleshooting
- Development guide with testing and contribution guidelines
- Security policy with vulnerability reporting process
- Setup validation script for new users

---

**Note**: This is the initial public release of Second Opinion. The project was developed as a professional-grade tool for AI model comparison and cost optimization.
