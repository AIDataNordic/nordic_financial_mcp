# Nordic Financial MCP

> [!WARNING]
> **Scheduled maintenance — ongoing (April 2026)**
> The `nordic_company_data` collection is currently being re-ingested with an enriched payload schema and upgraded embedding model (`intfloat/e5-large-v2`, 1024d) with hybrid dense+sparse search. Search results may be incomplete until ingestion is finished.

<!-- mcp-name: io.github.AIDataNordic/nordic-financial-mcp -->

A production-grade semantic search server for Nordic financial markets — built for autonomous AI agents. 375,000+ vectors across exchange filings, company reports, macro data, and press releases.

**Search:** Natural language queries over annual reports, quarterly reports, exchange announcements and macroeconomic summaries — filtered by company, ticker, country, sector or year. Two-stage hybrid retrieval (dense + sparse BM25, fused via RRF) with cross-encoder reranking for high-precision results.

**Live endpoint:** `https://mcp.aidatanorge.no/mcp`  
**Transport:** `streamable-http`  
**Registry:** [MCP Registry](https://registry.modelcontextprotocol.io) · [Glama.ai](https://glama.ai) · [mcp.so](https://mcp.so)

---

## Connect

Add to your MCP client config:

```json
{
  "mcpServers": {
    "nordic-financial": {
      "type": "streamable-http",
      "url": "https://mcp.aidatanorge.no/mcp"
    }
  }
}
```

Or with Claude Code:
```bash
claude mcp add --transport http nordic-financial https://mcp.aidatanorge.no/mcp
```

---

## Quick Test

**Try the live demo in your browser:**  
👉 [https://mcp.aidatanorge.no/demo](https://mcp.aidatanorge.no/demo)

No installation, no configuration. Just search for "Equinor dividend", "Swedish policy rate", or "salmon price Q3".

---

## For MCP Client Developers

This server follows the **StreamableHTTP** MCP transport. A complete handshake is required before calling tools.

### Full Handshake Example (Copy-Paste Ready)

```bash
# 1. Create session and capture session ID
SESSION_ID=$(curl -X GET https://mcp.aidatanorge.no/mcp \
  -H "Accept: application/json, text/event-stream" \
  -s -i | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')

echo "Session ID: $SESSION_ID"

# 2. Initialize session
curl -X POST https://mcp.aidatanorge.no/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "example-client", "version": "1.0"}
    }
  }'

# 3. Send initialized notification
curl -X POST https://mcp.aidatanorge.no/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc": "2.0", "method": "notifications/initialized"}'

# 4. List available tools
curl -X POST https://mcp.aidatanorge.no/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'

# 5. Perform a search
curl -X POST https://mcp.aidatanorge.no/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "search_filings",
      "arguments": {"query": "Equinor dividend", "limit": 3}
    }
  }'
```

### Common Issues & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `406 Not Acceptable` | Missing `text/event-stream` in Accept header | Send: `Accept: application/json, text/event-stream` |
| `400 Bad Request: Missing session ID` | No session established | First `GET /mcp`, use returned `mcp-session-id` header |
| `-32602 Invalid request parameters` | Missing `initialize` before `tools/list` | Complete steps 1-3 in order |

### Python Example with MCP SDK

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("https://mcp.aidatanorge.no/mcp") as transport:
    async with ClientSession(*transport) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool(
            "search_filings",
            {"query": "Norwegian housing market Q3 2024", "country": "NO"}
        )
        print(result.content[0].text)
```

### Why This Matters

This handshake is **automatic** in MCP-compliant clients like Claude Desktop, LangChain, and the MCP Python SDK. If you're building a custom client, following the sequence above ensures compatibility.

The `/demo` endpoint shows how a browser can perform the same handshake using JavaScript `fetch()` — view source for a working implementation.

---

## What This Is

AIDataNorge is a full-stack data pipeline and semantic search system that ingests, processes, and indexes financial data from Nordic markets into a vector database optimized for AI agent queries. It exposes data through a Model Context Protocol (MCP) server, making it natively compatible with Claude, LangChain, and other LLM-based agents.

The system is designed with autonomous machine-to-machine consumption in mind, including support for emerging agent payment protocols. The database is updated nightly.

---

## MCP Tools

### `search_filings`

Semantic search over Nordic company filings, press releases and macroeconomic summaries.

```python
search_filings(
    query="Nordea net interest margin outlook 2025",
    report_type="quarterly_report",  # annual_report | quarterly_report | press_release | macro_summary
    country="SE",                    # NO | SE | DK | FI
    ticker="NDA",                    # optional — filter by company ticker
    fiscal_year=2025,                # optional — filter by year
    sector="energy",                 # optional — seafood | energy | shipping
    limit=10                         # default 5, max 20
)
# Returns semantically ranked text chunks with rerank_score, hybrid_score, vector_score,
# company, ticker, country, fiscal_year, report_type, filing_date and full text.
```

**Search pipeline:** Dense embedding (`intfloat/e5-large-v2`, 1024d) + sparse BM25, fused via Reciprocal Rank Fusion (RRF), reranked by `mmarco-mMiniLMv2-L12-H384-v1`. Natural language queries in any language are supported.

### `get_company_info`

Look up a company in the official business registry.

```python
get_company_info(
    identifier="923609016",  # org/CVR/business ID
    country="NO"             # NO (Brønnøysund) | DK (CVR) | FI (PRH)
)
# Returns company name, status and registered address.
```

### `parse_pdf_to_text`

Download a PDF from a URL and extract all text, page by page.

```python
parse_pdf_to_text(
    pdf_url="https://example.com/annual_report_2024.pdf"
)
# Returns extracted text with page separators.
# Useful for reading report attachments not indexed in the main database.
```

### `get_current_power_price`

Real-time day-ahead electricity spot prices for all Nordic bidding zones.

```python
get_current_power_price(
    zone="NO1",              # NO1–NO5, SE1–SE4, DK1, DK2, FI
    include_tomorrow=False   # fetch tomorrow's prices if available (published ~13:00 CET)
)
# Returns EUR/kWh — current hour price + full hourly breakdown + daily min/max/avg.
# Norwegian zones sourced from hvakosterstrommen.no, others directly from ENTSO-E.
# Handles both PT60M (hourly) and PT15M (15-min) resolutions.
```

### `ping`

```python
ping(name="world")
# Returns: "Hello world! Nordic MCP server is running."
```

---

## Data Coverage

| Source | Geography | Content | Volume |
|--------|-----------|---------|--------|
| XBRL ESEF (filings.xbrl.org) | NO/SE/DK/FI/IS | Annual reports, regulated markets, 2020–present | ~100k vectors |
| MFN Nordics | SE/NO/DK/FI | Annual & quarterly reports, First North companies | ~116k vectors |
| Oslo Børs Newsweb | NO | Exchange announcements, 2020–present | ~83k vectors |
| Nasdaq Copenhagen | DK | Exchange announcements, 2020–present | growing |
| Cision | SE/NO/DK/FI | Press releases | ~20k vectors |
| GlobeNewswire | NO/SE/DK/FI | Press releases, updated hourly Mon–Fri | ~500 vectors |
| Macro Norway | Norway | GDP, CPI, rates, housing, salmon, power | 24 quarters |
| Macro Nordics | SE/DK/FI | Rates, housing, credit, power | 72 quarters |

**Total: 375,000+ vectors** · Updated nightly

---

## Architecture

```
Data Sources                 Pipeline                  Serving
─────────────────            ─────────────────         ─────────────────
XBRL ESEF               →    Python ingest scripts  →  Qdrant
MFN Nordics             →    + Playwright scraping  →  Vector Database
Oslo Børs Newsweb       →    + PDF extraction        →  (375,000+ vectors)
Nasdaq Copenhagen       →    + Chunking              →        ↓
Cision / GlobeNewswire  →
SSB / Norges Bank       →    + Chunking              →        ↓
SSB / Norges Bank       →    + Dense embeddings      →  MCP Server
SCB / DST / stat.fi     →
                        →      (e5-large-v2, 1024d)  →  (FastMCP 3.2)
                        →    + Sparse BM25            →        ↓
                        →    + RRF fusion             →  AI Agents / LLMs
```

---

## Technical Stack

**Data ingestion**
- Python with Playwright for JavaScript-rendered IR pages and MFN feed
- PyMuPDF (fitz) for PDF text extraction
- Paragraph-aware chunking (512-token chunks, 100-token overlap)
- Dense embeddings: `intfloat/e5-large-v2` (1024d)
- Sparse embeddings: `Qdrant/bm25` via fastembed

**Storage & search**
- Qdrant vector database (self-hosted)
- Hybrid dense+sparse retrieval with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking (`mmarco-mMiniLMv2-L12-H384-v1`)

**Serving**
- FastMCP 3.2 over HTTP (`/mcp` endpoint)
- Cloudflare Tunnel — rate limited to 60 req/min per IP
- Compatible with Claude, LangChain, and any MCP-capable agent

**Infrastructure**
- Ubuntu Server 24 LTS, self-hosted
- 16 GB RAM
- Automated cron jobs for continuous ingestion
- Bitcoin full node (LND) for Lightning Network payments
- DigiByte full node with DigiRail and DigiDollar Oracle node

---

## Agent Payment Infrastructure

The system is built with autonomous agent monetization in mind, supporting three complementary payment protocols:

**x402 Micropayments**  
A pay-per-call variant of the server (`mcp_server_x402.py`) is implemented using the [x402 protocol](https://x402.org) — the HTTP 402 payment standard for autonomous agents. Agents receive a payment requirement response, pay in USDC on Base, and retry automatically. Currently **paused** — x402 functionality will be integrated directly into the main server (`mcp_server.py`) in a future release.

**Lightning Network (L402)**  
Running a full Bitcoin node with LND enables L402 — the HTTP payment protocol for autonomous agents. Agents can discover the API, receive a Lightning invoice, pay in millisatoshis, and get access — all without human intervention. Infrastructure in place, monetization layer in development.

**DigiRail / DigiDollar**  
Also running a DigiByte full node with DigiRail (an agent payment protocol similar to L402) and a DigiDollar Oracle node. DigiDollar is the world's first UTXO-native decentralized stablecoin, implemented directly in DigiByte Core v9.26. The oracle node contributes to the decentralized price feed that maintains DigiDollar's USD peg — 15 of 30 randomly selected oracle nodes must reach consensus every ~25 minutes using Schnorr signatures.

This multi-protocol payment infrastructure (x402/Base + Bitcoin/Lightning + DigiByte/DigiRail) positions AIDataNorge to serve agents operating across different payment ecosystems.

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
| 03:17 Sundays | XBRL annual reports |
| 06:00 Mon–Fri | yfinance — stock prices and FX rates |
| 06:15 daily | MFN Nordics — quarterly reports and press releases |
| 06:30 Mon–Fri | ENTSO-E — energy data |
| 07:00 daily | Oslo Børs Newsweb — exchange announcements |
| 08:00–18:00 hourly Mon–Fri | GlobeNewswire — press releases (NO/SE/DK/FI) |
| 09:00 daily | Query analysis report (email) |

---

## Monitoring & Activity

### Check server health

```bash
# Qdrant responding?
curl http://localhost:6333

# Vector count
curl http://localhost:6333/collections/nordic_company_data | python3 -m json.tool
```

### Check MCP server process

```bash
ps aux | grep mcp_server.py
```

### Check MCP query activity

```bash
# Tail live log
tail -f ~/logs/mcp_server.log

# Run full query analysis
cd ~/norsk-mcp-server && venv/bin/python3 analyze_queries.py
```

### Check Cloudflare tunnel

```bash
journalctl -u cloudflared --since "1 hour ago" | tail -50
```

---

## Skills Demonstrated

- **RAG system design** — end-to-end pipeline from raw data to semantic search
- **Hybrid retrieval** — dense+sparse embeddings with RRF fusion and cross-encoder reranking
- **Web scraping at scale** — Playwright, RSS feeds, REST APIs, PDF extraction
- **Vector database operations** — Qdrant, embedding models, reranking
- **MCP server development** — FastMCP, tool design for LLM agents
- **Agent payment protocols** — x402, L402, DigiRail
- **Linux server administration** — process management, cron, systemd
- **Blockchain infrastructure** — Bitcoin full node + LND, DigiByte full node + oracle
- **Python engineering** — async pipelines, error handling, idempotent design
- **Financial data domain knowledge** — Nordic exchanges, regulatory filings, macro data

---

## Status (April 2026)

- `nordic_company_data`: 375,000+ vectors — XBRL, MFN, Newsweb, Cision, GlobeNewswire, macro
- MCP server: live at `https://mcp.aidatanorge.no/mcp`
- Published: MCP Registry · Glama.ai · mcp.so
- x402 pay-per-call: implemented, currently paused — will be integrated into main server
- L402 / DigiRail: infrastructure in place, monetization layer in development
- Live demo: `https://mcp.aidatanorge.no/demo`
