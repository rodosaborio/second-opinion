# Security Policy

## Supported Versions

We take security seriously and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within Second Opinion, please help us maintain the security of the project by reporting it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by:

1. **Opening a private vulnerability report** on GitHub:
   - Go to the [Security tab](https://github.com/rodosaborio/second-opinion/security) of this repository
   - Click "Report a vulnerability"
   - Fill out the form with details about the vulnerability

2. **Creating a private issue** if GitHub security advisories are not available:
   - Contact the maintainer directly through GitHub

### What to Include

When reporting a vulnerability, please include:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** of the vulnerability
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up questions

### Security Best Practices for Users

To use Second Opinion securely:

1. **Protect your API keys**:
   - Never commit `.env` files to version control
   - Use different encryption keys for database and sessions
   - Rotate API keys regularly

2. **Monitor costs and usage**:
   - Set conservative cost limits initially
   - Monitor usage through your API provider dashboards
   - Use the built-in cost protection features

3. **Keep dependencies updated**:
   - Regularly update to the latest version
   - Monitor security advisories for dependencies

4. **Validate inputs**:
   - Be cautious with untrusted input
   - Use input sanitization features (enabled by default)

## Security Features

Second Opinion includes several built-in security features:

- **Input sanitization** to prevent prompt injection attacks
- **Response filtering** to prevent API key leakage
- **Cost protection** to prevent runaway spending
- **Rate limiting** to prevent abuse
- **Data encryption** for local storage
- **API key validation** and secure storage

## Acknowledgments

We appreciate security researchers and users who report vulnerabilities responsibly. Contributors who report valid security issues will be acknowledged (with their permission) in release notes.

---

**Remember**: Security is a shared responsibility. Please report issues responsibly and help us keep Second Opinion secure for everyone.
