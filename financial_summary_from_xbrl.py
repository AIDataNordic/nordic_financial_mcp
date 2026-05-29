#!/usr/bin/env python3
"""
financial_summary_from_xbrl.py — Build financial_summary chunks from structured
XBRL fields already stored in xbrl_esef chunk payloads.

No re-download, no regex. Reads xbrl_esef annual report chunks,
picks best (ticker, fiscal_year) candidate and upserts one financial_summary chunk.

Usage:
    python financial_summary_from_xbrl.py --dry-run
    python financial_summary_from_xbrl.py --ticker NHY --dry-run
    python financial_summary_from_xbrl.py
"""

import argparse
import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, PointStruct
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTION = "nordic_company_data"
BATCH_SIZE = 500
EMBED_PREFIX = "passage: "

# Reporting currency by country (default — some companies report in EUR regardless)
COUNTRY_CURRENCY = {"NO": "NOK", "SE": "SEK", "DK": "DKK", "FI": "EUR", "IS": "ISK"}

# Structured fields to read from payload, in display order
KPI_FIELDS = [
    ("revenue_eur",             "Revenue"),
    ("operating_profit_eur",    "Operating profit"),
    ("net_profit_eur",          "Net profit"),
    ("total_assets_eur",        "Total assets"),
    ("total_equity_eur",        "Total equity"),
    ("total_liabilities_eur",   "Total liabilities"),
    ("current_assets_eur",      "Current assets"),
    ("noncurrent_assets_eur",   "Non-current assets"),
    ("current_liabilities_eur", "Current liabilities"),
    ("noncurrent_liabilities_eur", "Non-current liabilities"),
    ("cash_eur",                "Cash"),
    ("borrowings_eur",          "Borrowings"),
    ("receivables_eur",         "Receivables"),
    ("inventories_eur",         "Inventories"),
]

client = QdrantClient(host="localhost", port=6333, timeout=60)
dense_model = SentenceTransformer("intfloat/e5-large-v2", device="cpu")
sparse_model = SparseTextEmbedding("Qdrant/bm25")


def count_kpis(payload: dict) -> int:
    return sum(1 for f, _ in KPI_FIELDS if payload.get(f) is not None)


def build_text(company: str, ticker: str, fiscal_year: int, currency: str, payload: dict) -> str:
    parts = [f"{company} ({ticker}) FY{fiscal_year} — annual financial summary ({currency} millions):"]
    for field, label in KPI_FIELDS:
        val = payload.get(field)
        if val is not None:
            parts.append(f"{label}: {currency} {val:,.0f}M")
    employees = payload.get("employee_count")
    if employees:
        parts.append(f"Employees: {int(employees):,}")
    return "\n".join(parts)


def embed(text: str) -> dict:
    dense = dense_model.encode(EMBED_PREFIX + text, normalize_embeddings=True).tolist()
    sp = list(sparse_model.embed([text]))[0]
    return {"dense": dense, "sparse": {"indices": sp.indices.tolist(), "values": sp.values.tolist()}}


def load_existing() -> set[tuple]:
    existing = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="report_type", match=MatchValue(value="financial_summary")),
                FieldCondition(key="source", match=MatchValue(value="extracted_xbrl")),
            ]),
            limit=1000, offset=offset,
            with_payload=["ticker", "fiscal_year"], with_vectors=False,
        )
        for r in records:
            p = r.payload or {}
            existing.add((p.get("ticker"), p.get("fiscal_year")))
        if offset is None:
            break
    log.info(f"Loaded {len(existing)} existing financial_summary records.")
    return existing


def run(dry_run: bool, ticker_filter: Optional[str], force: bool = False):
    existing = load_existing() if not force else set()

    must = [
        FieldCondition(key="source", match=MatchValue(value="xbrl_esef")),
        FieldCondition(key="report_type", match=MatchValue(value="annual_report")),
    ]
    if ticker_filter:
        must.append(FieldCondition(key="ticker", match=MatchValue(value=ticker_filter)))

    # Accumulate best candidate per (ticker, fiscal_year) — most KPIs wins
    candidates: dict[tuple, dict] = {}

    offset = None
    scrolled = 0
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=must),
            limit=BATCH_SIZE, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not records:
            break

        for rec in records:
            p = rec.payload or {}
            ticker = p.get("ticker") or ""
            fiscal_year = p.get("fiscal_year")
            if not ticker or not fiscal_year:
                continue
            if count_kpis(p) == 0:
                continue

            key = (ticker, fiscal_year)
            if key not in candidates or count_kpis(p) > count_kpis(candidates[key]):
                candidates[key] = p

        scrolled += len(records)
        log.info(f"Scrolled {scrolled} XBRL chunks | candidates: {len(candidates)}")
        if offset is None:
            break

    log.info(f"Done. {len(candidates)} unique (ticker, fiscal_year) with XBRL data.")

    to_upsert = []
    skipped = 0

    for (ticker, fiscal_year), p in candidates.items():
        if (ticker, fiscal_year) in existing:
            skipped += 1
            continue

        company = p.get("company_name") or ticker
        country = p.get("country") or ""
        currency = COUNTRY_CURRENCY.get(country, "EUR")

        text = build_text(company, ticker, fiscal_year, currency, p)
        log.info(f"  {'[DRY] ' if dry_run else ''}→ {text.splitlines()[0]}")
        if dry_run:
            print(text)
            print()

        if not dry_run:
            vectors = embed(text)
            to_upsert.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload={
                    "report_type":        "financial_summary",
                    "source":             "extracted_xbrl",
                    "company_name":       company,
                    "ticker":             ticker,
                    "fiscal_year":        fiscal_year,
                    "country":            country,
                    "market":             p.get("market"),
                    "currency":           currency,
                    "unit":               "million",
                    "revenue":            p.get("revenue_eur"),
                    "operating_profit":   p.get("operating_profit_eur"),
                    "net_profit":         p.get("net_profit_eur"),
                    "total_assets":       p.get("total_assets_eur"),
                    "total_equity":       p.get("total_equity_eur"),
                    "total_liabilities":  p.get("total_liabilities_eur"),
                    "current_assets":     p.get("current_assets_eur"),
                    "current_liabilities": p.get("current_liabilities_eur"),
                    "cash":               p.get("cash_eur"),
                    "borrowings":         p.get("borrowings_eur"),
                    "employee_count":     p.get("employee_count"),
                    "text":               text,
                },
            ))

        if not dry_run and len(to_upsert) >= 50:
            client.upsert(COLLECTION, to_upsert)
            log.info(f"Upserted batch of {len(to_upsert)}")
            to_upsert = []

    if not dry_run and to_upsert:
        client.upsert(COLLECTION, to_upsert)
        log.info(f"Upserted final batch of {len(to_upsert)}")

    action = "Would upsert" if dry_run else "Upserted"
    log.info(f"{action}: {len(candidates) - skipped} | Skipped (already exists): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--force", action="store_true", help="Re-upsert all, overwrite existing")
    args = parser.parse_args()
    run(dry_run=args.dry_run, ticker_filter=args.ticker, force=args.force)
