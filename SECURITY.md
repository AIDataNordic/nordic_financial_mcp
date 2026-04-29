# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities via public GitHub issues.

Email: **hallvardo@gmail.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 48 hours and aim to release a fix within 7 days of confirmation.

## Scope

This server exposes a public read-only MCP endpoint. There is no authentication layer and no user data is stored. The main attack surface is:
- Prompt injection via search results returned to AI clients
- Resource exhaustion via high-frequency requests (rate limiting is handled at the Cloudflare layer)
