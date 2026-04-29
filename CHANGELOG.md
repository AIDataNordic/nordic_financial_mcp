# Changelog

## [1.0.1] — 2026-04-28

### Changed
- Bumped `fastmcp` 3.2.0 → 3.2.4
- Bumped `sentence-transformers` 5.3.0 → 5.4.1
- Bumped `fastapi` 0.135.2 → 0.136.1
- Bumped `fastembed` 0.7.4 → 0.8.0
- Bumped `pymupdf` 1.24.0 → 1.27.2.3

## [1.0.0] — 2026-04-28

### Added
- Initial public release
- `search_filings` — hybrid semantic search (dense e5-large-v2 + sparse BM25, RRF fusion, cross-encoder reranking) over 570 000+ Nordic financial documents
- `get_company_info` — live business registry lookup for Norway (Brønnøysund), Denmark (CVR) and Finland (PRH)
- `get_current_power_price` — real-time day-ahead spot prices for all 11 Nordic bidding zones (NO1–NO5, SE1–SE4, DK1, DK2, FI)
- `parse_pdf_to_text` — download and extract text from public PDFs
- `ping` — connectivity check
- 3 MCP prompts: `company_analysis`, `power_price_analysis`, `macro_outlook`
- Streamable-HTTP transport via FastMCP 3.2
- Dockerfile and CI workflow
- Registered on Smithery, Glama and MCP Registry
