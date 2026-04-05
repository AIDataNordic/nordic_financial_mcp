<!-- mcp-name: io.github.AIDataNordic/nordic-financial-mcp -->

# AIDataNorge

A production-grade AI data infrastructure for Nordic financial markets — built from scratch on a self-hosted Linux server. Designed to serve autonomous AI agents with high-quality, semantically searchable data about Nordic publicly listed companies, macroeconomics, and regulatory filings.

---

## What This Is

AIDataNorge is a full-stack data pipeline and semantic search system that ingests, processes, and indexes financial data from Nordic markets into a vector database optimized for AI agent queries. It exposes data through a Model Context Protocol (MCP) server, making it natively compatible with Claude, LangChain, and other LLM-based agents.

The system is designed with autonomous machine-to-machine consumption in mind, including support for emerging agent payment protocols.

---

## Architecture

```
Data Sources                 Pipeline                  Serving
─────────────────            ─────────────────         ─────────────────
Oslo Børs (NewsWeb)    →                               
SEC EDGAR (20-F/6-K)   →     Python ingest scripts  →  Qdrant
MFN Nordics (SE/DK/FI) →     + Playwright scraping  →  Vector Database
GlobeNewswire          →     + PDF extraction        →  (87,000+ vectors)
SSB / Norges Bank      →     + Chunking              →        ↓
SCB / DST / stat.fi    →     + Embeddings            →  MCP Server
ENTSO-E (power prices) →      (all-mpnet-base-v2)   →  (FastMCP 3.2)
IR websites (PDF)      →                                      ↓
                                                       AI Agents / LLMs
```

---

## Data Coverage

| Source | Geography | Content | Volume |
|--------|-----------|---------|--------|
| NewsWeb | Norway | Exchange filings 2020– | ~30,000+ docs |
| MFN Nordics | SE / DK / FI | Annual & quarterly reports | Growing |
| GlobeNewswire | NO/SE/DK/FI | Press releases | ~8,600 docs |
| SEC EDGAR | Nordic ADRs | 20-F / 6-K filings | Ongoing |
| IR websites | SE/DK/FI | Annual/quarterly PDFs | ~3,000 docs |
| Macro NO | Norway | GDP, CPI, rates, housing | 24 quarters |
| Macro Nordics | SE/DK/FI | Rates, housing, credit, power | In progress |

Total: **87,000+ vectors** across Nordic Large/Mid Cap companies (162 companies)

---

## Technical Stack

**Data ingestion**
- Python with Playwright for JavaScript-rendered IR pages and MFN feed
- PyMuPDF (fitz) for PDF text extraction
- Paragraph-aware chunking (512-token chunks, 100-token overlap)
- Batch embedding with `sentence-transformers/all-mpnet-base-v2`

**Storage & search**
- Qdrant vector database (self-hosted)
- Cosine similarity search
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)

**Serving**
- FastMCP 3.2 over HTTP (`/mcp` endpoint)
- Tools exposed: `search_filings`, `get_company_info`, `get_market_data`
- Compatible with Claude, LangChain, and any MCP-capable agent

**Infrastructure**
- Ubuntu Server 24 LTS, self-hosted
- 14 GB RAM, ~950 GB storage (LVM)
- Automated cron jobs for continuous ingestion
- Bitcoin full node (LND) for Lightning Network payments
- DigiByte full node with DigiRail and DigiDollar Oracle node

---

## Agent Payment Infrastructure

The system is built with autonomous agent monetization in mind:

**Lightning Network (L402)**
Running a full Bitcoin node with LND enables L402 — the HTTP payment protocol for autonomous agents. Agents can discover the API, receive a Lightning invoice, pay in millisatoshis, and get access — all without human intervention.

**DigiRail / DigiDollar**
Also running a DigiByte full node with DigiRail (an agent payment protocol similar to L402) and a DigiDollar Oracle node. DigiDollar is the world's first UTXO-native decentralized stablecoin, implemented directly in DigiByte Core v9.26. The oracle node contributes to the decentralized price feed that maintains DigiDollar's USD peg — 15 of 30 randomly selected oracle nodes must reach consensus every ~25 minutes using Schnorr signatures.

This dual payment infrastructure (Bitcoin/Lightning + DigiByte/DigiRail) positions AIDataNorge to serve agents operating across different payment ecosystems.

---

## MCP Tools

```python
search_filings(
    query="Nordea net interest margin outlook 2025",
    report_type="quarterly_report",  # or annual_report, macro_summary, press_release
    country="SE",                    # NO, SE, DK, FI
    limit=10
)
# Returns semantically ranked chunks with reranking, company metadata, and source URL

get_company_info(org_number)
# Norwegian company lookup via Brønnøysundregistrene

get_market_data(ticker)
# Live price and key ratios via Yahoo Finance
```

---

## Ingest Pipeline Design

Each data source has a dedicated ingest script with:
- Idempotent processing via MD5-based point IDs (upsert-safe)
- `processed.txt` log to avoid redundant re-fetching
- `nohup` + cron scheduling for unattended overnight runs
- Structured payload per chunk: `source`, `country`, `ticker`, `company_name`, `report_type`, `published_date`, `chunk_index`, `total_chunks`

Chunking strategy: paragraphs are accumulated until reaching the 512-token model window. Chunks never split mid-sentence. 100-token overlap ensures context continuity across chunk boundaries.

---

## Cron Schedule

| Time | Job |
|------|-----|
| 07:00 daily | NewsWeb update (Oslo Børs) |
| 08:00–18:00 hourly (Mon–Fri) | GlobeNewswire (NO/SE/DK/FI) |
| Quarterly | Macro Norway (SSB + Norges Bank) |
| Quarterly (pending) | Macro Nordics (SCB/DST/stat.fi + ENTSO-E) |

---

## Skills Demonstrated

- **RAG system design** — end-to-end pipeline from raw data to semantic search
- **Web scraping at scale** — Playwright, RSS feeds, REST APIs, PDF extraction
- **Vector database operations** — Qdrant, embedding models, reranking
- **MCP server development** — FastMCP, tool design for LLM agents
- **Linux server administration** — LVM, process management, cron, nohup
- **Blockchain infrastructure** — Bitcoin full node + LND, DigiByte full node + oracle
- **Python engineering** — async pipelines, error handling, idempotent design
- **Financial data domain knowledge** — Nordic exchanges, regulatory filings, macro data

---

## Status (April 2026)

- NewsWeb backfill complete: 500,000 → 669,999 (6,083 filings)
- MFN Nordics ingest running: 162 Large/Mid Cap companies (SE/DK/FI)
- Macro Norway complete: 2020Q1–2025Q4
- Macro Nordics: script ready, pending ENTSO-E token
- MCP server: live on port 8003
- L402 / DigiRail: infrastructure in place, monetization layer in development

---

*Built and operated by a single developer as a passion project exploring the intersection of Nordic financial data, AI agents, and decentralized payment infrastructure.*
# nordic_financial_mcp
