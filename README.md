# Nordic Financial MCP

> [!WARNING]
> **Scheduled maintenance — 16. april 2026**
> The `nordic_company_data` collection is currently being re-ingested from scratch.
> Semantic search via MCP is temporarily unavailable. Expected completion: 12. april 2026.
> Reason: Upgrading embedding model from `all-mpnet-base-v2` (768d) to `intfloat/e5-large-v2` (1024d) with hybrid dense+sparse search.

<!-- mcp-name: io.github.AIDataNordic/nordic-financial-mcp -->

A production-grade semantic search server for Nordic financial markets — built for autonomous AI agents. 173,000+ vectors across exchange filings, company reports, macro data, and commodity prices.

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
  -H "Content-Type": application/json" \
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

**Search quality:** Two-stage retrieval — dense vector search (`all-mpnet-base-v2`) followed by cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for high-precision results.

---

## Data Coverage

| Source | Geography | Content | Volume |
|--------|-----------|---------|--------|
| NewsWeb | Norway | Exchange filings 2020– | ~30,000+ docs |
| MFN Nordics | SE / DK / FI | Annual & quarterly reports, 162 companies | ~90,000+ docs |
| GlobeNewswire | NO/SE/DK/FI | Press releases, updated hourly | ~8,600 docs |
| SEC EDGAR | Nordic ADRs | 20-F / 6-K filings | Ongoing |
| IR websites | SE/DK/FI | Annual/quarterly PDFs | ~3,000 docs |
| Macro NO | Norway | GDP, CPI, rates, housing | 24 quarters |
| Macro Nordics | SE/DK/FI | Rates, housing, credit, power | 72 quarters |
| Commodity | Global/Nordic | Brent, salmon, and more | Per quarter |

**Total: 173,000+ vectors** · Updated nightly

---

## Architecture

```
Data Sources                 Pipeline                  Serving
─────────────────            ─────────────────         ─────────────────
Oslo Børs (NewsWeb)    →                               
SEC EDGAR (20-F/6-K)   →     Python ingest scripts  →  Qdrant
MFN Nordics (SE/DK/FI) →     + Playwright scraping  →  Vector Database
GlobeNewswire          →     + PDF extraction        →  (173,000+ vectors)
SSB / Norges Bank      →     + Chunking              →        ↓
SCB / DST / stat.fi    →     + Embeddings            →  MCP Server
ENTSO-E (power prices) →      (all-mpnet-base-v2)   →  (FastMCP 3.2)
IR websites (PDF)      →                                      ↓
                                                       AI Agents / LLMs
```

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
- Cloudflare Tunnel — rate limited to 60 req/min per IP
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

## Monitoring & Activity

### Check server health

```bash
# Qdrant running?
docker ps | grep qdrant

# Qdrant responding?
curl http://localhost:6333

# Vector count (main collection)
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

# Run full query analysis (per day, per hour, top queries, latency)
cd ~/norsk-mcp-server && venv/bin/python3 analyze_queries.py
```

`analyze_queries.py` parses `~/logs/mcp_server.log` and reports:
- Total `search_filings` calls
- Traffic per day and per hour (UTC)
- Most common search queries
- Breakdown by country and report type
- Search latency (avg / min / max)
- Zero-result queries

### Check Cloudflare tunnel

```bash
# Tunnel status and connections
journalctl -u cloudflared --since "1 hour ago" | tail -50
```

For HTTP request analytics (geographic distribution, status codes, request volume), log in to [dash.cloudflare.com](https://dash.cloudflare.com) → select domain → **Analytics & Logs → Traffic**.

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

- NewsWeb backfill complete: 500,000 → 669,999
- MFN Nordics: 162 Large/Mid Cap companies (SE/DK/FI)
- Macro Norway complete: 2020Q1–2025Q4
- Macro Nordics complete: SE/DK/FI 2020Q1–2025Q4
- MCP server: live at `https://mcp.aidatanorge.no/mcp`
- Published: MCP Registry · Glama.ai · mcp.so
- GitHub topics: `mcp` `mcp-server` `nordic` `finance` `semantic-search` `qdrant` `norway` `sweden` `denmark` `finland`
- L402 / DigiRail: infrastructure in place, monetization layer in development
- Live demo: `https://mcp.aidatanorge.no/demo`

---

*Built and operated by a single developer as a passion project exploring the intersection of Nordic financial data, AI agents, and decentralized payment infrastructure.*
