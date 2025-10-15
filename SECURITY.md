# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of The Convergence seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by emailing:
- **Email:** aria@persistos.co
- **Subject:** [SECURITY] Brief description of issue

### What to Include

Please include the following information:
- Type of vulnerability
- Affected components/versions
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial Response:** Within 48 hours
- **Status Update:** Within 7 days
- **Fix Timeline:** Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days  
  - Medium: Within 30 days
  - Low: Next release cycle

### Disclosure Policy

- We will acknowledge receipt of your report
- We will investigate and validate the issue
- We will develop and test a fix
- We will release a patch and security advisory
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices for Users

### API Key Management

**Never commit API keys to version control:**

```bash
# ✅ Good - Use environment variables
export OPENAI_API_KEY="sk-..."
convergence optimize config.yaml

# ❌ Bad - Hardcoded in YAML
# api:
#   auth:
#     token: "sk-hardcoded-key"  # DON'T DO THIS
```

**Be cautious with custom endpoints:**
- Verify SSL certificates
- Use trusted DNS providers
- Consider using a VPN for sensitive operations

### Data Privacy

**Sanitize test cases:**

```json
{
  "test_cases": [
    {
      "input": {
        "prompt": "Generic test prompt"  // ✅ No PII
      },
      "expected": {}
    }
  ]
}
```

**Clean results before sharing:**

```bash
# Remove sensitive responses before committing examples
jq 'del(.all_results[].test_results[].response_text)' results/detailed_results.json
```

### Dependency Security

**Keep dependencies updated:**

```bash
# Check for security updates
pip list --outdated

# Update safely
pip install --upgrade the-convergence

# Audit dependencies
pip-audit  # Install with: pip install pip-audit
```

### Storage Security

**Protect local databases:**

```bash
# Legacy database contains optimization history
chmod 600 data/legacy.db

# Backup and encrypt sensitive data
tar czf backup.tar.gz data/ | gpg -e -r your@email.com
```

### Docker Security (if applicable)

**Use secure base images:**

```dockerfile
FROM python:3.11-slim  # Use official images
# Don't use: FROM random/python-image
```

**Don't expose unnecessary ports:**

```yaml
# docker-compose.yml
services:
  convergence:
    # Don't expose database ports externally
    # Only expose what's needed
```

## Known Security Considerations

### 1. LLM Provider Access

The Convergence makes API calls to external LLM providers (OpenAI, Anthropic, etc.). Be aware:

- API keys grant access to paid services
- Providers may log requests (check their policies)
- Responses may contain generated content
- Rate limits may apply

**Mitigation:**
- Use separate API keys for testing
- Monitor API usage and costs
- Review provider security policies
- Consider using mock mode for development

### 2. File System Access

The framework writes files to disk (results, cache, databases):

- File storage can consume disk space
- Cached data persists between runs
- Legacy database grows over time

**Mitigation:**
- Set appropriate file permissions
- Implement disk quotas if needed
- Regularly clean old results
- Backup important data

### 3. Code Execution

Custom evaluators are Python code that gets executed:

```python
# examples/my_api/my_evaluator.py
def score_response(result, expected, params, metric=None):
    # This code executes during optimization
    ...
```

**Mitigation:**
- Only use trusted evaluators
- Review custom code before running
- Use code review for team evaluators
- Consider sandboxing in production

### 4. Database Injection

While SQLite queries use parameterization, be cautious with:

- Custom storage backends
- User-provided key names
- Complex query patterns

**Mitigation:**
- Use built-in storage backends
- Validate input keys
- Avoid complex custom queries

## Vulnerability History

### v0.1.0 (Current)
- No known vulnerabilities
- Initial release security review complete

## External Dependencies

We rely on these security-critical dependencies:

- **httpx:** HTTP client (handles SSL/TLS)
- **pydantic:** Input validation
- **litellm:** LLM provider abstraction
- **aiosqlite:** Async SQLite access

**Security Monitoring:**
- Dependabot enabled (future)
- Regular security audits planned
- CVE monitoring in progress

## Security Checklist for Contributors

When contributing code, ensure:

- [ ] No hardcoded secrets or API keys
- [ ] Input validation on all user-provided data
- [ ] Parameterized database queries
- [ ] HTTPS for external API calls
- [ ] Proper error handling (no information leakage)
- [ ] File path sanitization
- [ ] Type hints for security-critical functions
- [ ] Tests for security-related features

## Security Update Process

1. Vulnerability reported or discovered
2. Issue triaged and severity assessed
3. Fix developed and tested
4. Security advisory published
5. Patched version released
6. Users notified via:
   - GitHub Security Advisory
   - Release notes
   - Email (for critical issues)

## Questions?

For security-related questions (non-vulnerabilities):
- Open a GitHub Discussion with [Security] tag
- Email: aria@persistos.co

For security vulnerabilities:
- **Always email directly:** aria@persistos.co
- Include [SECURITY] in subject line

---

**Last Updated:** October 15, 2025
**Version:** 1.0
**Status:** Active
