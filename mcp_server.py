"""
Nordic MCP Server - fastmcp v3
Semantic search over Nordic company filings, press releases and macroeconomic
summaries, with hybrid dense+sparse retrieval and cross-encoder reranking.
"""

import os
import logging
import time
import json
import torch
import httpx
from datetime import datetime
from typing import Annotated
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from fastapi.responses import HTMLResponse
from mcp.types import ToolAnnotations

# --- Configuration ---
COLLECTION_NAME  = "nordic_company_data"
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
RERANK_FETCH     = 20
RERANK_MODEL     = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print("Loading embedding model...")
_model = SentenceTransformer("intfloat/e5-large-v2", device="cpu")
_model.max_seq_length = 512
print("Embedding model loaded.")

print("Loading sparse model...")
_sparse_model = SparseTextEmbedding("Qdrant/bm25")
print("Sparse model loaded.")

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL, device="cpu")
print("Reranker loaded.")

_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Logging ---
_log = logging.getLogger("mcp")
_log.setLevel(logging.INFO)
os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)
_fh = logging.FileHandler(os.path.expanduser("~/logs/mcp_server.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
_log.addHandler(_fh)

mcp = FastMCP("nordic-public-data-mcp")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def ping(name: Annotated[str, "Name to include in the greeting"] = "world") -> str:
    """Simple connectivity test. Returns a greeting to confirm the server is running."""
    _log.info(f'ping name="{name}"')
    return f"Hello {name}! Nordic MCP server is running."


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_company_info(
    identifier: Annotated[str, "Organisation number (NO: 9 digits, DK: 8 digits CVR, FI: business ID with hyphen)"],
    country: Annotated[str, "Two-letter country code: NO (default), DK, or FI"] = "NO",
) -> dict:
    """Look up a company in the official business registry for Norway, Denmark or Finland.

    Args:
        identifier: Organisation/business/CVR number. Format varies by country:
                    NO: 9-digit organisation number, e.g. 923609016 (Equinor)
                    DK: 8-digit CVR number, e.g. 22756214 (Maersk)
                    FI: Business ID with hyphen, e.g. 0112038-9 (Nokia)
        country:    Two-letter country code: 'NO' (default), 'DK', or 'FI'.

    Returns:
        Dict with company name, status and registered business address.
    """
    async with httpx.AsyncClient() as client:
        try:
            if country.upper() == "NO":
                url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{identifier}"
                resp = await client.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "name":    data.get("navn"),
                    "status":  data.get("status"),
                    "address": data.get("forretningsadresse", {}).get("adresse"),
                    "country": "NO",
                }

            elif country.upper() == "FI":
                url = f"https://avoindata.prh.fi/opendata-ytj-api/v3/companies?businessId={identifier}"
                resp = await client.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                companies = data.get("companies", [])
                if not companies:
                    return {"error": f"No company found for businessId {identifier}"}
                co = companies[0]
                names = co.get("names", [])
                active_name = next((n["name"] for n in names if not n.get("endDate")), names[0]["name"] if names else None)
                addresses = co.get("addresses", [])
                active_addr = next((a for a in addresses if not a.get("endDate")), None)
                return {
                    "name":       active_name,
                    "businessId": co.get("businessId", {}).get("value"),
                    "address":    active_addr.get("street") if active_addr else None,
                    "city":       active_addr.get("city") if active_addr else None,
                    "country":    "FI",
                }

            elif country.upper() == "DK":
                url = f"https://cvrapi.dk/api?vat={identifier}&country=dk"
                resp = await client.get(url, timeout=10, headers={"User-Agent": "nordic-mcp-server hallvardo@gmail.com"})
                data = resp.json()
                if "error" in data:
                    return {"error": data["error"]}
                return {
                    "name":    data.get("name"),
                    "status":  None if not data.get("enddate") else "ophørt",
                    "address": f"{data.get('address')}, {data.get('zipcode')} {data.get('city')}",
                    "country": "DK",
                }

            else:
                return {"error": f"Country '{country}' not supported. Use NO, DK or FI."}

        except Exception as e:
            return {"error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def search_filings(
    query: Annotated[str, "Natural language search query, e.g. 'Equinor dividend 2024' or 'Norwegian housing market Q3'"],
    ticker: Annotated[str, "Filter by company ticker, e.g. 'EQNR', 'SALM', 'NDA'"] = "",
    fiscal_year: Annotated[int, "Filter by fiscal year, e.g. 2024. Use 0 for no filter"] = 0,
    report_type: Annotated[str, "Filter by type: annual_report, quarterly_report, press_release, exchange_announcement, macro_summary"] = "",
    sector: Annotated[str, "Filter by sector, e.g. 'energy', 'financials', 'salmon'"] = "",
    country: Annotated[str, "Filter by country: NO, SE, DK, or FI"] = "",
    limit: Annotated[int, "Number of results to return (1–20)"] = 5,
) -> list[dict]:
    """Search the Nordic financial database for company filings, press releases
    and macroeconomic summaries.

    The database contains ~375 000 vectors across four Nordic markets (NO/SE/DK/FI).

    COMPANY FILINGS
      Annual reports (XBRL/ESEF) and quarterly reports from ~1 500 listed companies
      across Oslo Børs, Nasdaq Stockholm, Nasdaq Helsinki, Nasdaq Copenhagen and
      First North markets. Covers 2020–present. Strong coverage for NO and SE;
      growing coverage for DK and FI.

    EXCHANGE ANNOUNCEMENTS & PRESS RELEASES
      Regulatory filings, exchange announcements and press releases from listed
      companies in NO, SE, DK and FI. Covers 2020–present.

    MACROECONOMIC SUMMARIES
      Quarterly macro summaries covering key indicators per country:
        Norway (NO):  policy rate, FX rates, CPI, house prices, credit growth,
                      electricity price, salmon price, GDP components
        Sweden (SE):  policy rate, house price index, household credit
        Denmark (DK): policy rate, house price index, household loans,
                      electricity price
        Finland (FI): house price index, household debt-to-income ratio,
                      electricity price
      Use report_type='macro_summary' and country='NO'/'SE'/'DK'/'FI' to filter.
      Use fiscal_year and a quarter reference in your query, e.g.
      "Norwegian housing market Q1 2024".

    Args:
        query:       What you are looking for, e.g. 'net interest margin outlook',
                     'salmon price Q3', 'dividend policy', 'fleet utilization',
                     'Norwegian housing market 2024 Q1',
                     'Swedish policy rate inflation 2023'
        ticker:      Optional — filter by company ticker, e.g. 'SALM', 'EQNR', 'NDA'
        fiscal_year: Optional — filter by year, e.g. 2024
        report_type: Optional — one of:
                         'annual_report'     – Nordic XBRL/ESEF annual reports
                         'quarterly_report'  – Quarterly/interim reports
                         'press_release'     – Exchange announcements and press releases
                         'macro_summary'     – Quarterly macroeconomic summaries
        sector:      Optional — filter by sector:
                         'seafood'   – seafood companies
                         'energy'    – energy / oil & gas
                         'shipping'  – shipping companies
        country:     Optional — filter by country code: 'NO', 'SE', 'DK' or 'FI'
        limit:       Number of results after reranking (default 5, max 20)

    Returns:
        List of relevant text excerpts with metadata, reranked by relevance.
        Each result includes rerank_score, hybrid_score, vector_score, company,
        ticker, country, fiscal_year, report_type, period, filing_date and the
        full text chunk.
    """
    limit = min(limit, 20)
    _t0 = time.time()

    # e5-large-v2 requires "query:"-prefix at search time
    e5_query = f"query: {query}"

    with torch.no_grad():
        dense_vec = _model.encode(e5_query, normalize_embeddings=True).tolist()

    sparse_result = list(_sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    conditions = []
    if ticker:
        conditions.append(
            FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
        )
    if fiscal_year:
        conditions.append(
            FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))
        )
    if report_type:
        conditions.append(
            FieldCondition(key="report_type", match=MatchValue(value=report_type))
        )
    if sector:
        conditions.append(
            FieldCondition(key="sector", match=MatchValue(value=sector.lower()))
        )
    if country:
        conditions.append(
            FieldCondition(key="country", match=MatchValue(value=country.upper()))
        )

    query_filter = Filter(must=conditions) if conditions else None

    fetch_limit = max(RERANK_FETCH, limit * 4)
    try:
        results = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec,   using="dense",  limit=fetch_limit),
                Prefetch(query=sparse_vec,  using="sparse", limit=fetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
        )
    except Exception as e:
        _log.exception(f"Qdrant query failed: {e}")
        return []

    if not results.points:
        return []

    candidates = results.points
    pairs = [(query, p.payload.get("text", "")) for p in candidates]
    with torch.no_grad():
        rerank_scores = _reranker.predict(pairs)

    # Hybrid scoring: vector_score boosts rerank multiplicatively — can tip ties but can't rescue weak reranks
    r_scores = [float(s) for s in rerank_scores]
    v_scores = [p.score for p in candidates]
    v_max = max(v_scores) or 1e-8

    hybrid_scores = [
        r * (1.0 + 0.3 * (v / v_max))
        for r, v in zip(r_scores, v_scores)
    ]

    ranked = sorted(
        zip(hybrid_scores, r_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    output = []
    for hybrid_score, rerank_score, point in ranked:
        if len(output) >= limit:
            break
        # Discard cross-encoder false positives: high rerank score but near-zero vector similarity
        if rerank_score > 7.0 and point.score < 0.05:
            continue
        p = point.payload
        is_macro = p.get("report_type") == "macro"
        output.append({
            "rerank_score":  round(float(rerank_score), 4),
            "hybrid_score":  round(float(hybrid_score), 4),
            "vector_score":  round(point.score, 4),
            "company":       p.get("macro_label") if is_macro else p.get("company_name"),
            "ticker":        p.get("macro_symbol") if is_macro else p.get("ticker"),
            "sector":        p.get("macro_category") if is_macro else p.get("sector"),
            "country":       p.get("country"),
            "fiscal_year":   p.get("fiscal_year"),
            "report_type":   p.get("report_type"),
            "period":        p.get("period") or p.get("period_ending"),
            "filing_date":   p.get("filing_date") or p.get("published_date"),
            "text":          p.get("text"),
            "chunk_index":   p.get("chunk_index"),
            "total_chunks":  p.get("total_chunks"),
        })

    elapsed = round(time.time() - _t0, 3)
    _log.info(f'search_filings query="{query}" ticker="{ticker}" report_type="{report_type}" country="{country}" results={len(output)} elapsed={elapsed}s')
    return output


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def parse_pdf_to_text(pdf_url: Annotated[str, "Direct HTTPS URL to the PDF file"]) -> str:
    """Download a PDF from a URL and extract all text as a single string, page by page.
    
    This is useful for agents that need to read report attachments, press releases,
    or any PDF content that is not directly searchable in the main database.
    
    Args:
        pdf_url: Direct URL to the PDF file (e.g. https://example.com/report.pdf)
    
    Returns:
        All text from the PDF with page separators, or an error message.
    """
    import aiohttp
    import fitz  # PyMuPDF
    
    _log.info(f'parse_pdf_to_text url="{pdf_url}"')
    pdf_document = None
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                if response.status != 200:
                    error_msg = f"Download failed: HTTP {response.status}"
                    _log.error(f'parse_pdf_to_text {error_msg}')
                    return error_msg
                
                pdf_bytes = await response.read()
        
        if b"%PDF" not in pdf_bytes[:10]:
            error_msg = "URL did not return a PDF (got HTML or other content)"
            _log.error(f'parse_pdf_to_text {error_msg}')
            return error_msg
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = []
        num_pages = len(pdf_document)
        
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            if text.strip():
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
            else:
                all_text.append(f"--- Page {page_num + 1} (no extractable text) ---")
        
        if not all_text:
            return "PDF contains no extractable text (may be a scanned image PDF)."
        
        result = "\n\n".join(all_text)
        _log.info(f'parse_pdf_to_text success url="{pdf_url}" pages={num_pages} chars={len(result)}')
        return result
    
    except aiohttp.ClientError as e:
        error_msg = f"Network error downloading PDF: {str(e)}"
        _log.error(f'parse_pdf_to_text {error_msg}')
        return error_msg
    except Exception as e:
        error_msg = f"PDF parsing error: {str(e)}"
        _log.error(f'parse_pdf_to_text {error_msg}')
        return error_msg
    finally:
        if pdf_document is not None:
            pdf_document.close()


_ENTSOE_KEY = os.getenv("ENTSOE_API_KEY")

_ZONE_EIC = {
    "NO1": "10YNO-1--------2", "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J", "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "SE1": "10Y1001A1001A44P", "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L", "SE4": "10Y1001A1001A47J",
    "FI":  "10YFI-1--------U",
    "DK1": "10YDK-1--------W", "DK2": "10YDK-2--------M",
}

_NO_ZONES = {"NO1", "NO2", "NO3", "NO4", "NO5"}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_current_power_price(
    zone: Annotated[str, "Bidding zone: NO1–NO5, SE1–SE4, DK1, DK2, or FI"] = "NO1",
    include_tomorrow: Annotated[bool, "Also fetch tomorrow's prices if available (published after 13:00 CET)"] = False,
) -> dict:
    """Fetch today's hourly electricity spot prices for a Nordic bidding zone.

    All zones return prices in EUR/kWh (NordPool native currency).
    Tomorrow's prices are typically available after 13:00 CET.

    Args:
        zone:             Bidding zone: NO1 (East), NO2 (Southwest), NO3 (Central),
                          NO4 (North), NO5 (West), SE1–SE4, DK1, DK2, FI.
        include_tomorrow: Also fetch tomorrow's prices if available (default False).

    Returns:
        Dict with current hour price, today's hourly prices, and summary stats.
    """
    import xml.etree.ElementTree as ET
    from datetime import date, datetime, timedelta, timezone

    zone = zone.upper()
    if zone not in _ZONE_EIC:
        return {"error": f"Unknown zone '{zone}'. Valid zones: {sorted(_ZONE_EIC)}"}

    today = date.today()

    async def fetch_no(d: date) -> list[dict] | None:
        url = (f"https://www.hvakosterstrommen.no/api/v1/prices"
               f"/{d.year}/{d.month:02d}-{d.day:02d}_{zone}.json")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            raw = resp.json()
        return [
            {"time": h["time_start"][11:16],
             "EUR_per_kWh": round(h["EUR_per_kWh"], 5)}
            for h in raw
        ]

    async def fetch_entsoe(d: date) -> list[dict] | None:
        if not _ENTSOE_KEY:
            return None
        eic = _ZONE_EIC[zone]
        period_start = d.strftime("%Y%m%d") + "0000"
        period_end   = (d + timedelta(days=1)).strftime("%Y%m%d") + "0000"
        params = {
            "securityToken": _ENTSOE_KEY,
            "documentType": "A44",
            "in_Domain": eic, "out_Domain": eic,
            "periodStart": period_start, "periodEnd": period_end,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://web-api.tp.entsoe.eu/api", params=params)
            if resp.status_code != 200 or "<Acknowledgement" in resp.text:
                return None
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        root = ET.fromstring(resp.text)
        # Collect raw points keyed by UTC datetime
        raw: dict[datetime, list[float]] = {}
        for ts in root.findall(".//ns:TimeSeries", ns):
            resolution = ts.findtext(".//ns:resolution", namespaces=ns)
            if resolution == "PT60M":
                step = timedelta(hours=1)
            elif resolution == "PT15M":
                step = timedelta(minutes=15)
            else:
                continue
            start_str = ts.findtext(".//ns:timeInterval/ns:start", namespaces=ns)
            if not start_str:
                continue
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            for pt in ts.findall(".//ns:Point", ns):
                pos   = int(pt.findtext("ns:position", namespaces=ns))
                price = float(pt.findtext("ns:price.amount", namespaces=ns))
                dt    = start_dt + (pos - 1) * step
                # Round down to hour for aggregation
                hour_dt = dt.replace(minute=0, second=0, microsecond=0)
                raw.setdefault(hour_dt, []).append(price)
        if not raw:
            return None
        return [
            {"time": dt.strftime("%H:%M"),
             "EUR_per_kWh": round(sum(prices) / len(prices) / 1000, 5)}
            for dt, prices in sorted(raw.items())
        ]

    def summarize(hours: list[dict]) -> dict:
        prices = [h["EUR_per_kWh"] for h in hours]
        result = {
            "min_EUR_per_kWh": round(min(prices), 5),
            "max_EUR_per_kWh": round(max(prices), 5),
            "avg_EUR_per_kWh": round(sum(prices) / len(prices), 5),
            "hours": hours,
        }
        return result

    fetch = fetch_no if zone in _NO_ZONES else fetch_entsoe
    today_data = await fetch(today)
    if not today_data:
        return {"error": f"No price data available for zone {zone} on {today}"}

    now_h = datetime.now(timezone.utc).hour
    current = next(
        (h for h in today_data if int(h["time"][:2]) == now_h),
        today_data[0],
    )

    result = {
        "zone": zone,
        "date": today.isoformat(),
        "current_hour_utc": f"{now_h:02d}:00",
        "current": current,
        "today": summarize(today_data),
    }

    if include_tomorrow:
        tomorrow_data = await fetch(today + timedelta(days=1))
        result["tomorrow"] = summarize(tomorrow_data) if tomorrow_data else "not yet available"

    _log.info(f'get_current_power_price zone="{zone}" current={current["EUR_per_kWh"]} EUR/kWh')
    return result


# --- DEMO ENDEPUNKT (nettleser-demo) ---
DEMO_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Nordisk Finanssøk - Demo</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        input, button { padding: 10px; font-size: 16px; }
        input { width: 70%; }
        button { cursor: pointer; background: #0066cc; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0052a3; }
        .result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #f9f9f9; }
        .result strong { color: #0066cc; }
        .loading { color: #666; font-style: italic; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>🔍 Nordisk Finanssøk</h1>
    <p>Søk i 130 000+ nordiske finansdokumenter, pressemeldinger og makrodata</p>
    
    <input type="text" id="query" placeholder="F.eks. 'Equinor dividend' eller 'norsk boligpris Q3 2024'" style="width: 70%">
    <button onclick="search()">Søk</button>
    
    <div id="results"></div>

    <script>
        const MCP_URL = window.location.origin + '/mcp';
        
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">🔍 Søker etter "' + query + '"...</div>';
            
            try {
                const sessionRes = await fetch(MCP_URL, {
                    method: 'GET',
                    headers: { 'Accept': 'application/json, text/event-stream' }
                });
                const sessionId = sessionRes.headers.get('mcp-session-id');
                
                if (!sessionId) throw new Error('Kunne ikke opprette session');
                
                await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({
                        jsonrpc: "2.0", id: 1, method: "initialize",
                        params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "web-demo", version: "1.0" } }
                    })
                });
                
                await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })
                });
                
                const searchRes = await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({
                        jsonrpc: "2.0", id: 2, method: "tools/call",
                        params: { name: "search_filings", arguments: { query: query, limit: 5 } }
                    })
                });
                
                const text = await searchRes.text();
                const lines = text.split('\\n');
                let data = null;
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        data = JSON.parse(line.substring(6));
                        break;
                    }
                }
                
                if (data?.error) {
                    throw new Error(data.error.message || 'Ukjent feil');
                }
                
                if (data?.result?.content) {
                    const results = JSON.parse(data.result.content[0].text);
                    displayResults(results);
                } else {
                    resultsDiv.innerHTML = '<div class="error">Ingen resultater funnet</div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error">❌ Feil: ' + error.message + '</div>';
                console.error(error);
            }
        }
        
        function displayResults(results) {
            const container = document.getElementById('results');
            if (!results.length) {
                container.innerHTML = '<div class="error">Ingen resultater</div>';
                return;
            }
            
            container.innerHTML = results.map(r => `
                <div class="result">
                    <strong>${escapeHtml(r.company || 'Ukjent')}</strong> 
                    ${r.ticker ? '(' + r.ticker + ')' : ''} 
                    ${r.report_type ? '- ' + r.report_type.replace('_', ' ') : ''}<br>
                    <small>📊 Relevans: ${r.rerank_score.toFixed(2)} | 📅 ${r.fiscal_year || 'Ukjent år'}${r.country ? ' | 🌍 ' + r.country : ''}</small>
                    <p>${escapeHtml(r.text.substring(0, 400))}${r.text.length > 400 ? '...' : ''}</p>
                </div>
            `).join('');
        }
        
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>'''


@mcp.prompt()
def company_analysis(ticker: str, year: int = 2024) -> str:
    """Generate a prompt for analysing a Nordic listed company."""
    return (
        f"Use search_filings to find annual reports, quarterly reports and press releases "
        f"for {ticker} from {year}. Summarise revenue, operating profit, key risks and outlook. "
        f"Also check get_company_info for registered details."
    )


@mcp.prompt()
def power_price_analysis(zone: str = "NO1") -> str:
    """Generate a prompt for analysing Nordic electricity prices."""
    return (
        f"Use get_current_power_price for zone {zone} to get today's spot prices. "
        f"Then use search_filings with report_type='macro_summary' to find recent energy market context. "
        f"Summarise current price level, hourly pattern and relevant macro drivers."
    )


@mcp.prompt()
def macro_outlook(country: str = "NO") -> str:
    """Generate a prompt for a Nordic country macro overview."""
    return (
        f"Use search_filings with country='{country}' and report_type='macro_summary' "
        f"to retrieve the latest macro summaries. Cover policy rate, inflation, housing market "
        f"and GDP growth. Highlight any notable trends or risks."
    )


@mcp.custom_route("/demo", methods=["GET"])
async def demo_endpoint(request):
    """Demo side for nettleser"""
    return HTMLResponse(content=DEMO_HTML)


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT") == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.getenv("MCP_PORT", 8003))
        print(f"→ Starting MCP server at http://0.0.0.0:{port}/mcp")
        print(f"→ Demo available at http://0.0.0.0:{port}/demo")
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
