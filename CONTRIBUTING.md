# Contributing

Contributions are welcome — bug reports, data-source suggestions and pull requests.

## Issues

Use GitHub Issues to report bugs or suggest new data sources. Please include:
- What you searched for / which tool you called
- The response you got
- What you expected instead

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Make your changes. Keep commits focused and atomic.
3. Run a quick syntax check: `python -m py_compile mcp_server.py`
4. Open a PR with a clear description of what changed and why.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python mcp_server.py
```

The server requires a running Qdrant instance at `localhost:6333`. Set environment variables per `.env.example` (create one from the variables listed in README).

## Code Style

- Follow existing patterns in `mcp_server.py`.
- No unnecessary abstractions — keep it simple.
- No AI-generated docblock boilerplate.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
