"""
alfred.py — Nordic Due Diligence MCP Server

Retrieval orchestrator for Nordic company due diligence. Agents call one tool:
due_diligence_report(company). Alfred handles search strategy internally.

Phase 1: Haiku plans financial/operational/competitor/news sections (~$0.02-0.05)
Phase 2: Alfred derives macro + power sections from identified financial periods
Phase 3: Parallel searches via upstream Nordic Financial MCP (localhost:8003)
Phase 4: Post-process — detect power contracts, label spot prices accordingly

Port: 8006
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Annotated

import anthropic
from dotenv import load_dotenv
from fastmcp import Client, FastMCP

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("alfred")

UPSTREAM_URL = os.getenv("ALFRED_UPSTREAM_URL", "http://localhost:8003/mcp")

# ---------------------------------------------------------------------------
# Macro factor mappings
# ---------------------------------------------------------------------------

SECTOR_MACRO_DEFAULTS: dict[str, list[str]] = {
    "aluminium":              ["ALI=F", "NOKUSD=X"],
    "metals":                 ["ALI=F", "NOKUSD=X"],
    "aquaculture":            ["salmon_price", "NOKUSD=X"],
    "oil_gas":                ["BZ=F", "NOKUSD=X"],
    "oil_services":           ["BZ=F", "NOKUSD=X"],
    "shipping_tanker":        ["BZ=F", "NOKUSD=X"],
    "shipping_dry_bulk":      ["NOKUSD=X"],
    "industrials":            ["SEKUSD=X"],
    "industrials_distribution": ["SEKUSD=X"],
    "banking":                ["NOKUSD=X", "SEKUSD=X"],
    "real_estate":            ["NOKUSD=X"],
    "telecom":                [],
    "technology":             [],
}

MACRO_QUERY_TEMPLATES: dict[str, str] = {
    "ALI=F":        "aluminium price ALI=F {period}",
    "BZ=F":         "Brent crude oil price BZ=F {period}",
    "NG=F":         "natural gas price NG=F {period}",
    "NOKUSD=X":     "NOK USD exchange rate NOKUSD=X {period}",
    "SEKUSD=X":     "SEK USD exchange rate SEKUSD=X {period}",
    "DKKUSD=X":     "DKK USD exchange rate DKKUSD=X {period}",
    "salmon_price": "salmon spot price NOK per kg Atlantic {period}",
}

QUARTER_MONTHS: dict[str, str] = {
    "Q1": "January February March",
    "Q2": "April May June",
    "Q3": "July August September",
    "Q4": "October November December",
    "FY": "full year annual",
}

POWER_CONTRACT_SIGNALS = [
    "kraftavtale", "power agreement", "ppa", "realized power price",
    "hedged power", "power contract", "long-term power", "industrial power",
    "fastprisavtale", "langsiktig kraftavtale", "power purchase",
]

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Alfred",
    instructions=(
        "Alfred is a due diligence research assistant for Nordic listed companies "
        "(Norway, Sweden, Denmark, Finland, Iceland). "
        "Call due_diligence_report with a company name to receive structured data "
        "covering financials, risk factors, sector macro, and recent news. "
        "Alfred handles all search strategy internally — no query formulation needed. "
        "Your agent synthesizes the returned data into a report."
    ),
)

# ---------------------------------------------------------------------------
# Haiku planning prompt
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a due diligence research planner for Nordic listed companies.

Given a company name and today's date, output a JSON research plan. Output ONLY valid JSON — no markdown, no explanation.

WHAT YOU PLAN (sections only — Alfred handles macro and power separately):
- 2-3 financial period sections (newest first)
- 1-2 operational/segment sections specific to the company's business
- 1 risk section
- 1-2 competitor sections (embed competitor name in query, not ticker)
- 2-3 news sections by year, newest first (report_type: "press_release")
Total: 8-11 sections

WHAT YOU DECLARE (not sections):
- macro_factors: list chosen from available factors below
- power_zones: list of relevant zones if energy-intensive (else empty list)

AVAILABLE MACRO FACTORS (choose only from this list):
  ALI=F (aluminium price), BZ=F (Brent crude), NG=F (natural gas),
  NOKUSD=X (NOK/USD rate), SEKUSD=X (SEK/USD rate), DKKUSD=X (DKK/USD rate),
  salmon_price (Atlantic salmon spot price NOK/kg)

DETERMINING MOST RECENT DATA (use today's date):
- Q1 results: published ~late April/early May  → available if today >= May 1
- Q2 results: published ~late July/early August → available if today >= Aug 1
- Q3 results: published ~late October/early November → available if today >= Nov 1
- Q4/full-year results: published ~mid-February → available if today >= Feb 15
- Start with most recent available period, work backwards

SEARCH STRATEGY:
- Always embed the company name in every query string
- Set "company_filter": true for focal-company sections, false for competitors
- fiscal_year field: set to the relevant year for news sections to filter by year
- Do NOT plan macro or power sections — Alfred generates these automatically

COMPETITORS (embed full company name in query):
- Salmon/aquaculture: Mowi, SalMar, Lerøy, Austevoll, Bakkafrost
- Oil & gas: Equinor, Aker BP, Vår Energi
- Oil services: Subsea 7, Aker Solutions, Seadrill
- Aluminium/metals: Norsk Hydro, Elkem
- Shipping (tanker): Frontline, Hafnia, BW LPG
- Shipping (dry bulk): Golden Ocean, 2020 Bulkers
- Industrials/distribution: Lagercrantz, Indutrade, Addlife
- Telecom: Telia, Elisa, TDC
- Banking: DNB, Handelsbanken, Swedbank
- Pick the 2 closest competitors, limit 3 chunks each

POWER ZONES (energy-intensive sectors only — else empty list):
NO1=Oslo, NO2=Kristiansand, NO3=Trondheim, NO4=Tromsø, NO5=Bergen
SE1-SE4=Sweden, DK1=West Denmark, DK2=East Denmark, FI=Finland

OUTPUT FORMAT (example: today 2026-05-05, Mowi, Q1 2026 available):
{
  "ticker": "MOWI",
  "country": "NO",
  "sector": "aquaculture",
  "macro_factors": ["salmon_price", "NOKUSD=X"],
  "power_zones": [],
  "sections": [
    {"name": "financials_q1_2026", "query": "Mowi revenue EBIT harvest volume Q1 2026", "report_type": "", "fiscal_year": 0, "limit": 5, "company_filter": true},
    {"name": "financials_q4_2025", "query": "Mowi revenue EBIT harvest volume Q4 2025 full year results", "report_type": "", "fiscal_year": 0, "limit": 5, "company_filter": true},
    {"name": "financials_fy2025", "query": "Mowi revenue EBIT margins annual full year 2025", "report_type": "annual_report", "fiscal_year": 0, "limit": 5, "company_filter": true},
    {"name": "harvest_volume", "query": "Mowi harvest volume gutted weight guidance 2025 2026", "report_type": "", "fiscal_year": 0, "limit": 5, "company_filter": true},
    {"name": "risks", "query": "Mowi risk factors sea lice disease mortality regulatory 2025 2026", "report_type": "", "fiscal_year": 0, "limit": 5, "company_filter": true},
    {"name": "competitor_salmar", "query": "SalMar revenue harvest volume EBIT 2025 2026", "report_type": "", "fiscal_year": 0, "limit": 3, "company_filter": false},
    {"name": "competitor_lerøy", "query": "Lerøy revenue harvest volume EBIT 2025 2026", "report_type": "", "fiscal_year": 0, "limit": 3, "company_filter": false},
    {"name": "news_2026", "query": "Mowi acquisition guidance dividend outlook 2026", "report_type": "press_release", "fiscal_year": 2026, "limit": 3, "company_filter": true},
    {"name": "news_2025", "query": "Mowi acquisition dividend results strategy 2025", "report_type": "press_release", "fiscal_year": 2025, "limit": 3, "company_filter": true}
  ]
}"""


# ---------------------------------------------------------------------------
# Period extraction and section generation
# ---------------------------------------------------------------------------

def _extract_periods(sections: list[dict]) -> list[tuple[str, int]]:
    """Extract (period, year) from financial section names, ordered as planned."""
    seen = set()
    periods = []
    for s in sections:
        m = re.match(r"financials_(q[1-4]|fy)_(\d{4})", s.get("name", ""), re.IGNORECASE)
        if m:
            period = m.group(1).upper()
            year = int(m.group(2))
            key = (period, year)
            if key not in seen:
                seen.add(key)
                periods.append(key)
    return periods


def _generate_macro_sections(periods: list, macro_factors: list) -> list[dict]:
    """One macro section per period × factor, aligned with financial periods."""
    sections = []
    for period, year in periods:
        months = QUARTER_MONTHS.get(period, "")
        period_str = f"{months} {year}".strip()
        for factor in macro_factors:
            template = MACRO_QUERY_TEMPLATES.get(factor)
            if not template:
                continue
            safe = factor.replace("=", "").replace("/", "")
            sections.append({
                "name":           f"macro_{safe}_{period.lower()}_{year}",
                "query":          template.format(period=period_str),
                "report_type":    "macro",
                "fiscal_year":    year,
                "limit":          3,
                "company_filter": False,
            })
    return sections


def _generate_power_sections(periods: list, power_zones: list) -> list[dict]:
    """One power price section per period × zone, aligned with financial periods."""
    sections = []
    for period, year in periods:
        months = QUARTER_MONTHS.get(period, "")
        period_str = f"{months} {year}".strip()
        for zone in power_zones:
            sections.append({
                "name":           f"power_{zone.lower()}_{period.lower()}_{year}",
                "query":          f"day-ahead electricity price {zone} {period_str}",
                "report_type":    "macro",
                "zone_ticker":    zone,   # used as hard ticker filter in _build_args
                "limit":          2,
                "company_filter": False,
            })
    return sections


def _detect_power_contract(sections_output: dict) -> str:
    """Check power_costs chunks for contract signals. Returns 'contracted' or 'spot'."""
    power_text = " ".join(
        c.get("text", "")
        for c in sections_output.get("power_costs", [])
        + sections_output.get("power_electricity_costs", [])
    ).lower()
    if any(kw in power_text for kw in POWER_CONTRACT_SIGNALS):
        return "contracted"
    return "spot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def _probe_ticker(upstream: "Client", company: str) -> str | None:
    """Look up the actual ticker from Qdrant before planning."""
    try:
        result = await upstream.call_tool("search_filings", {"query": company, "limit": 5})
        rows = _parse_tool_result(result)
        tickers = [r.get("ticker") for r in rows if r.get("ticker")]
        if not tickers:
            return None
        return max(set(tickers), key=tickers.count)
    except Exception as e:
        _log.warning(f"Ticker probe failed for {company!r}: {e}")
        return None


async def _plan(company: str, confirmed_ticker: str | None = None) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    user_msg = f"Today is {datetime.utcnow().strftime('%Y-%m-%d')}. Plan due diligence for: {company}"
    if confirmed_ticker:
        user_msg += f"\n\nVerified ticker from database: {confirmed_ticker}"
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=_PLAN_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return json.loads(_strip_json(response.content[0].text))


def _parse_tool_result(result) -> list | dict:
    if not result:
        return []
    if hasattr(result, "structured_content"):
        sc = result.structured_content
        if isinstance(sc, dict):
            return sc.get("result", sc)
        return sc
    return []


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def due_diligence_report(
    company: Annotated[str, "Company name or ticker, e.g. 'Mowi', 'Equinor', 'Norsk Hydro', 'EQNR'"],
) -> dict:
    """Get comprehensive due diligence data for any Nordic listed company.

    Returns structured data grouped by section: financials (recent periods),
    operational metrics, risk factors, sector macro aligned to financial periods,
    power prices (historical, mapped to financial periods), competitors, and news.

    Alfred handles all search strategy internally. Your agent synthesizes the report.

    Args:
        company: Company name or ticker
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not configured."}

    async with Client(UPSTREAM_URL) as upstream:
        # Step 1: probe for confirmed ticker
        confirmed_ticker = await _probe_ticker(upstream, company)
        if confirmed_ticker:
            _log.info(f'alfred probe ticker="{confirmed_ticker}" for "{company}"')

        # Step 2: Haiku plans financial/operational/competitor/news sections
        try:
            plan = await _plan(company, confirmed_ticker)
        except Exception as e:
            _log.error(f"Planning failed for {company!r}: {e}")
            return {"error": f"Planning failed: {e}"}

        haiku_sections = plan.get("sections", [])
        power_zones    = plan.get("power_zones", [])
        sector         = plan.get("sector", "")

        # Step 3: derive macro factors (plan override or sector defaults)
        macro_factors = plan.get("macro_factors") or SECTOR_MACRO_DEFAULTS.get(sector, [])

        # Step 4: extract financial periods and generate aligned macro + power sections
        periods         = _extract_periods(haiku_sections)
        macro_sections  = _generate_macro_sections(periods, macro_factors)
        power_sections  = _generate_power_sections(periods, power_zones)

        all_sections = haiku_sections + macro_sections + power_sections

        _log.info(
            f'alfred company="{company}" ticker={plan.get("ticker")} sector={sector} '
            f'periods={periods} macro_factors={macro_factors} power_zones={power_zones} '
            f'sections={len(haiku_sections)}+{len(macro_sections)}macro+{len(power_sections)}power'
        )

        # Step 5: fire all searches in parallel
        def _build_args(s: dict) -> dict:
            args = {
                "query": s.get("query", ""),
                "limit": min(int(s.get("limit", 5)), 10),
            }
            if s.get("zone_ticker"):
                args["ticker"] = s["zone_ticker"]
            elif s.get("company_filter") and confirmed_ticker:
                args["ticker"] = confirmed_ticker
            rt = s.get("report_type", "")
            if rt:
                args["report_type"] = rt
            fy = int(s.get("fiscal_year") or 0)
            if fy and not s.get("zone_ticker"):  # ENTSO-E data has no fiscal_year
                args["fiscal_year"] = fy
            return args

        search_coros = [
            upstream.call_tool("search_filings", _build_args(s))
            for s in all_sections
        ]

        all_results = await asyncio.gather(*search_coros, return_exceptions=True)

    # Step 6: assemble output
    output_sections: dict = {}
    for s, result in zip(all_sections, all_results):
        if isinstance(result, Exception):
            _log.warning(f"Section {s['name']} failed: {result}")
            output_sections[s["name"]] = []
        else:
            output_sections[s["name"]] = _parse_tool_result(result)

    # Step 7: detect power contract from filing text
    spot_price_role = None
    if power_zones:
        spot_price_role = _detect_power_contract(output_sections)
        _log.info(f'alfred power role="{spot_price_role}" for "{company}"')

    return {
        "company":          company,
        "ticker_haiku":     plan.get("ticker", ""),
        "ticker_confirmed": confirmed_ticker,
        "country":          plan.get("country", ""),
        "sector":           sector,
        "macro_factors":    macro_factors,
        "power_zones":      power_zones,
        "spot_price_role":  spot_price_role,
        "periods_covered":  [f"{p}{y}" for p, y in periods],
        "generated_at":     datetime.utcnow().isoformat() + "Z",
        "sections":         output_sections,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.getenv("MCP_PORT", 8006))
        print(f"→ Starting Alfred at http://0.0.0.0:{port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
