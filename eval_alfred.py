"""
eval_alfred.py — Eval rammeverk for Alfred due diligence pipeline

Steg 1: Hent ground truth fra financial_summary-chunks i Qdrant
Steg 2: Kall Alfred for hvert selskap (HTTP til port 8006)
Steg 3: Score — finner Alfred korrekte finanstall i xbrl_financials-seksjonen?
Steg 4: Rapport — dekning, nøyaktighet, per-seksjon-kvalitet

Kjøring:
    venv/bin/python3 eval_alfred.py
    venv/bin/python3 eval_alfred.py --companies "Norsk Hydro" "Finnair" "AAK"
    venv/bin/python3 eval_alfred.py --auto 100
    venv/bin/python3 eval_alfred.py --dry-run
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from fastmcp import Client
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Filter, FieldCondition, MatchValue

ALFRED_URL = "http://localhost:8006/mcp"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "nordic_company_data"
REVENUE_TOLERANCE = 0.10  # ±10%

BAD_GROUND_TRUTH = {"EMGS", "ETTE", "KEMPOWR", "MOB", "OTEC"}  # kjente scale-feil / usikker data

# Kuratert liste: (display_name, ticker) — navn sendes til Alfred, ticker brukes for ground-truth-oppslag
GOLDEN_COMPANIES = [
    ("Norsk Hydro",    "NHY"),
    ("AAK",            "AAK"),
    ("Finnair",        "FIA1S"),
    ("Metso",          "OUKPY"),
    ("Valmet",         "VLMTY"),
    ("Gram Car Carriers", "GCCNOK"),
    ("Gyldendal",      "GYLDB"),
    ("Q-linea",        "QLINEA"),
    ("Cemat",          "CEMAT"),
    ("EMGS",           "EMGS"),
    ("Sivers Semiconductors", "SIVE"),
    ("Strongpoint",    "STRO"),
    ("Acrinova",       "ACRIB"),
    ("Image Systems",  "IS"),
    ("KABE",           "KABEB"),
]


@dataclass
class GroundTruth:
    company_name: str
    ticker: str
    fiscal_year: int
    revenue: float
    total_assets: Optional[float]
    total_equity: Optional[float]
    currency: str


@dataclass
class EvalResult:
    company_name: str
    ticker: str
    ground_truth: Optional[GroundTruth]
    alfred_elapsed: float = 0.0
    alfred_error: Optional[str] = None
    ticker_confirmed: Optional[str] = None
    sections_returned: list[str] = field(default_factory=list)
    xbrl_chunks: int = 0
    revenue_found: Optional[float] = None
    revenue_match: Optional[bool] = None   # None = not scoreable
    section_coverage: dict[str, int] = field(default_factory=dict)


FX_TO_EUR = {"EUR": 1.0, "NOK": 1/11, "SEK": 1/11, "DKK": 1/7.5, "ISK": 1/150}

def auto_select_companies(client: QdrantClient, n: int) -> list[tuple[str, str]]:
    """Fetch top N companies by revenue from financial_summary, spread across countries."""
    results, _ = client.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="source",      match=MatchValue(value="extracted_xbrl")),
            FieldCondition(key="report_type", match=MatchValue(value="financial_summary")),
        ]),
        limit=2000,
        with_payload=["ticker", "company_name", "fiscal_year", "revenue", "currency", "country"],
        with_vectors=False,
    )
    # Best (most recent with revenue) per ticker
    best: dict[str, dict] = {}
    for p in results:
        pay = p.payload
        t = pay.get("ticker")
        if not t or not pay.get("revenue") or t in BAD_GROUND_TRUTH:
            continue
        existing = best.get(t)
        if not existing or (pay.get("fiscal_year") or 0) > (existing.get("fiscal_year") or 0):
            best[t] = pay

    # Sort by revenue normalised to EUR, filter out scale-bugged values
    # (no real company has > 1 trillion EUR in revenue)
    MAX_REVENUE_EUR = 1_000_000  # 1 trillion EUR in millions

    def rev_eur(pay):
        fx = FX_TO_EUR.get(pay.get("currency", "EUR"), 1/10)
        return (pay.get("revenue") or 0) * fx

    by_country: dict[str, list] = {}
    for pay in best.values():
        if rev_eur(pay) > MAX_REVENUE_EUR:
            continue  # scale bug — skip
        c = pay.get("country", "??")
        by_country.setdefault(c, []).append(pay)
    for c in by_country:
        by_country[c].sort(key=rev_eur, reverse=True)

    # Round-robin across countries to ensure spread
    selected = []
    per_country = max(1, n // len(by_country))
    for c, rows in sorted(by_country.items()):
        selected.extend(rows[:per_country])
    # Top up to n if needed
    remaining = sorted(
        [p for c in by_country for p in by_country[c][per_country:]],
        key=rev_eur, reverse=True,
    )
    selected.extend(remaining[:max(0, n - len(selected))])
    selected = selected[:n]

    return [(p["company_name"], p["ticker"]) for p in selected]


def fetch_ground_truth(client: QdrantClient, ticker: str) -> Optional[GroundTruth]:
    """Fetch most recent financial_summary with revenue from Qdrant."""
    results, _ = client.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="source",      match=MatchValue(value="extracted_xbrl")),
            FieldCondition(key="report_type", match=MatchValue(value="financial_summary")),
            FieldCondition(key="ticker",      match=MatchValue(value=ticker)),
        ]),
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    candidates = [p for p in results if p.payload.get("revenue")]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.payload.get("fiscal_year", 0))
    pay = best.payload
    return GroundTruth(
        company_name=pay.get("company_name", ticker),
        ticker=ticker,
        fiscal_year=pay.get("fiscal_year"),
        revenue=pay["revenue"],
        total_assets=pay.get("total_assets"),
        total_equity=pay.get("total_equity"),
        currency=pay.get("currency", "?"),
    )


def _parse_revenue_from_text(text: str) -> Optional[float]:
    """Extract revenue figure (millions) from financial_summary text chunk."""
    # Pattern: "Revenue: NOK 207,929M" or "Revenue: EUR 4,863M"
    m = re.search(r"Revenue:\s+\w+\s+([\d,]+(?:\.\d+)?)M", text)
    if m:
        return float(m.group(1).replace(",", ""))
    # Fallback: bare tall etter "revenue" på samme linje
    m = re.search(r"[Rr]evenue[^:]*:\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _score_revenue(found: float, expected: float) -> bool:
    if expected == 0:
        return False
    return abs(found - expected) / abs(expected) <= REVENUE_TOLERANCE


def _section_coverage(sections: dict) -> dict[str, int]:
    return {k: len(v) for k, v in sections.items() if isinstance(v, list)}


async def call_alfred(company: str) -> dict:
    async with Client(ALFRED_URL) as client:
        result = await client.call_tool("due_diligence_report", {"company": company})
        if hasattr(result, "structured_content") and result.structured_content:
            sc = result.structured_content
            return sc.get("result", sc) if isinstance(sc, dict) else sc
        # Fallback: parse JSON from text content
        if hasattr(result, "content") and result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    return json.loads(block.text)
        return {}


async def eval_company(
    display_name: str,
    ticker: str,
    ground_truth: Optional[GroundTruth],
    dry_run: bool,
) -> EvalResult:
    result = EvalResult(
        company_name=display_name,
        ticker=ticker,
        ground_truth=ground_truth,
    )

    if dry_run:
        result.alfred_error = "dry-run"
        return result

    t0 = time.time()
    try:
        alfred_data = await call_alfred(display_name)
    except Exception as e:
        result.alfred_elapsed = round(time.time() - t0, 1)
        result.alfred_error = str(e)[:120]
        return result

    result.alfred_elapsed = round(time.time() - t0, 1)

    if "error" in alfred_data:
        result.alfred_error = alfred_data["error"]
        return result

    result.ticker_confirmed = alfred_data.get("ticker_confirmed")
    sections = alfred_data.get("sections", {})
    result.sections_returned = list(sections.keys())
    result.section_coverage = _section_coverage(sections)

    # Score xbrl_financials + xbrl_summary sections
    xbrl_chunks = sections.get("xbrl_financials", []) + sections.get("xbrl_summary", [])
    result.xbrl_chunks = len(xbrl_chunks)

    if ground_truth and ground_truth.revenue:
        for chunk in xbrl_chunks:
            chunk_fy = chunk.get("fiscal_year")
            if chunk_fy and chunk_fy != ground_truth.fiscal_year:
                continue  # wrong fiscal year — skip
            text = chunk.get("text", "")
            found = _parse_revenue_from_text(text)
            if found is not None:
                result.revenue_found = found
                result.revenue_match = _score_revenue(found, ground_truth.revenue)
                break
        if result.revenue_found is None:
            result.revenue_match = False  # no chunk matched correct fiscal year with parseable revenue

    return result


def print_report(results: list[EvalResult]) -> None:
    print("\n" + "=" * 80)
    print("ALFRED EVAL RAPPORT")
    print("=" * 80)

    revenue_scoreable = [r for r in results if r.revenue_match is not None]
    revenue_correct   = [r for r in revenue_scoreable if r.revenue_match]
    errored           = [r for r in results if r.alfred_error and r.alfred_error != "dry-run"]
    avg_elapsed       = sum(r.alfred_elapsed for r in results if not r.alfred_error) / max(1, len(results) - len(errored))

    print(f"\nSamlet: {len(results)} selskaper | "
          f"Feil: {len(errored)} | "
          f"Revenue-score: {len(revenue_correct)}/{len(revenue_scoreable)} "
          f"({100*len(revenue_correct)//max(1,len(revenue_scoreable))}%) | "
          f"Gj.sn. tid: {avg_elapsed:.0f}s")

    print(f"\n{'Selskap':<22} {'Ticker':<8} {'GT FY':<7} {'GT Rev':>12} "
          f"{'Alfred Rev':>12} {'Match':<6} {'xbrl#':<6} {'Tid':>5} {'Status'}")
    print("-" * 90)

    for r in results:
        gt = r.ground_truth
        gt_fy  = str(gt.fiscal_year) if gt else "-"
        gt_rev = f"{gt.revenue:,.0f}{gt.currency[:1]}" if gt and gt.revenue else "-"
        af_rev = f"{r.revenue_found:,.0f}" if r.revenue_found else "-"
        match  = "✓" if r.revenue_match else ("✗" if r.revenue_match is False else "n/a")
        status = r.alfred_error[:30] if r.alfred_error else "ok"
        print(f"{r.company_name:<22} {r.ticker:<8} {gt_fy:<7} {gt_rev:>12} "
              f"{af_rev:>12} {match:<6} {r.xbrl_chunks:<6} {r.alfred_elapsed:>4.0f}s  {status}")

    # Section coverage
    print("\n--- Seksjonsdekning (gj.sn. chunks per seksjon) ---")
    all_sections: dict[str, list[int]] = {}
    for r in results:
        for sec, count in r.section_coverage.items():
            all_sections.setdefault(sec, []).append(count)
    for sec, counts in sorted(all_sections.items()):
        avg = sum(counts) / len(counts)
        present = sum(1 for c in counts if c > 0)
        print(f"  {sec:<30} {avg:4.1f} chunks gj.sn  ({present}/{len(counts)} selskaper)")

    # Per-company section detail for failures
    failures = [r for r in results if r.revenue_match is False and not r.alfred_error]
    if failures:
        print("\n--- Detaljert seksjonsoversikt for mislykkede revenue-treff ---")
        for r in failures:
            print(f"\n  {r.company_name} ({r.ticker}): xbrl_financials={r.xbrl_chunks} chunks")
            print(f"    GT: FY{r.ground_truth.fiscal_year if r.ground_truth else '?'} "
                  f"rev={r.ground_truth.revenue if r.ground_truth else '?'} {r.ground_truth.currency if r.ground_truth else ''}")
            for sec, count in r.section_coverage.items():
                if "xbrl" in sec or "financials" in sec:
                    print(f"    {sec}: {count} chunks")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", help="Subset av selskapsnavn å evaluere")
    parser.add_argument("--auto", type=int, metavar="N", help="Hent topp N selskaper automatisk fra Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Hent ground truth men kall ikke Alfred")
    parser.add_argument("--output-json", help="Lagre råresultater til JSON-fil")
    args = parser.parse_args()

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    if args.auto:
        print(f"Auto-velger topp {args.auto} selskaper fra Qdrant...")
        companies = auto_select_companies(qdrant, args.auto)
        print(f"Valgte {len(companies)} selskaper fordelt på land.\n")
    elif args.companies:
        name_set = {n.lower() for n in args.companies}
        # Look up tickers from Qdrant financial_summary by company name
        all_summaries, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
            FieldCondition(key="source",      match=MatchValue(value="extracted_xbrl")),
            FieldCondition(key="report_type", match=MatchValue(value="financial_summary")),
        ]), limit=2000, with_payload=["ticker", "company_name"], with_vectors=False)
        name_to_ticker = {p.payload["company_name"].lower(): p.payload["ticker"]
                          for p in all_summaries if p.payload.get("company_name") and p.payload.get("ticker")}
        companies = []
        for name in args.companies:
            ticker = name_to_ticker.get(name.lower())
            if ticker:
                companies.append((name, ticker))
            else:
                print(f"Advarsel: '{name}' ikke funnet i Qdrant, hopper over.")
        if not companies:
            print("Ingen selskaper funnet.")
            sys.exit(1)
    else:
        companies = GOLDEN_COMPANIES

    print(f"Henter ground truth for {len(companies)} selskaper...")
    ground_truths = {}
    for name, ticker in companies:
        try:
            gt = fetch_ground_truth(qdrant, ticker)
        except (ResponseHandlingException, Exception) as e:
            gt = None
            print(f"  {ticker:<10} FEIL: {e!s:.60}")
            continue
        ground_truths[ticker] = gt
        status = f"FY{gt.fiscal_year} rev={gt.revenue:,.0f}{gt.currency[:1]}" if gt else "ingen revenue-data"
        print(f"  {ticker:<10} {status}")

    if args.dry_run:
        print("\n[dry-run] Hopper over Alfred-kall.")

    print(f"\nKaller Alfred for {len(companies)} selskaper{' (dry-run)' if args.dry_run else ''}...")
    results = []
    for i, (name, ticker) in enumerate(companies, 1):
        print(f"  [{i}/{len(companies)}] {name}...", end=" ", flush=True)
        r = await eval_company(name, ticker, ground_truths.get(ticker), args.dry_run)
        status = r.alfred_error or f"{r.alfred_elapsed:.0f}s, xbrl={r.xbrl_chunks}"
        print(status)
        results.append(r)

    print_report(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([
                {
                    "company": r.company_name,
                    "ticker": r.ticker,
                    "ticker_confirmed": r.ticker_confirmed,
                    "gt_fiscal_year": r.ground_truth.fiscal_year if r.ground_truth else None,
                    "gt_revenue": r.ground_truth.revenue if r.ground_truth else None,
                    "gt_currency": r.ground_truth.currency if r.ground_truth else None,
                    "revenue_found": r.revenue_found,
                    "revenue_match": r.revenue_match,
                    "xbrl_chunks": r.xbrl_chunks,
                    "alfred_elapsed": r.alfred_elapsed,
                    "alfred_error": r.alfred_error,
                    "section_coverage": r.section_coverage,
                }
                for r in results
            ], f, indent=2, ensure_ascii=False)
        print(f"\nRåresultater lagret til {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
