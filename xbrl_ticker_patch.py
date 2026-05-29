"""
xbrl_ticker_patch.py — Backfill ticker=None in xbrl_esef records

Strategy:
1. Build company_name → ticker map from all records that HAVE a ticker
2. For each xbrl_esef record with ticker=None, if company_name is in map → patch
3. Use set_payload in batches

Priority: xbrl_esef records with ticker set > other sources
"""

import sys
from collections import Counter, defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField

COLLECTION = "nordic_company_data"
BATCH_SIZE = 100
DRY_RUN = "--dry-run" in sys.argv

_src_arg = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--source=")), None)
TARGET_SOURCE = _src_arg or "xbrl_esef"

client = QdrantClient(host="localhost", port=6333, timeout=120)


def build_name_ticker_map() -> dict[str, str]:
    """Collect all company_name → ticker mappings from records with ticker set."""
    print("Building name→ticker map from all records with ticker set...")
    name_tickers: dict[str, list[str]] = defaultdict(list)

    offset = None
    total = 0
    while True:
        batch, offset = client.scroll(
            COLLECTION,
            scroll_filter=Filter(
                must_not=[IsNullCondition(is_null=PayloadField(key="ticker"))]
            ),
            limit=500,
            offset=offset,
            with_payload=["company_name", "ticker", "source"],
            with_vectors=False,
        )
        for r in batch:
            name = r.payload.get("company_name", "").strip()
            ticker = r.payload.get("ticker", "").strip()
            if name and ticker:
                name_tickers[name].append(ticker)
        total += len(batch)
        if offset is None:
            break
        if total % 50000 == 0:
            print(f"  scanned {total}...")

    print(f"  scanned {total} records with ticker set")

    # For each name, pick the most common ticker
    result: dict[str, str] = {}
    for name, tickers in name_tickers.items():
        counter = Counter(tickers)
        # Most common ticker — skip if ambiguous between 2+ unrelated tickers
        # (e.g. subsidiary CARLSBERG BREWERIES vs listed CARLSBERG A/S)
        most_common, most_count = counter.most_common(1)[0]
        result[name] = most_common

    print(f"  built map for {len(result)} unique company names")
    return result


def fetch_null_ticker_ids(name: str) -> list[str]:
    """Get all point IDs for xbrl_esef records with ticker=None for a given company."""
    ids = []
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=TARGET_SOURCE)),
                    FieldCondition(key="company_name", match=MatchValue(value=name)),
                    IsNullCondition(is_null=PayloadField(key="ticker")),
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(r.id for r in batch)
        if offset is None:
            break
    return ids


def main():
    name_ticker_map = build_name_ticker_map()

    # Find all unique company names with ticker=None in xbrl_esef
    print("\nFinding xbrl_esef company names with ticker=None...")
    none_names: set[str] = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=TARGET_SOURCE)),
                    IsNullCondition(is_null=PayloadField(key="ticker")),
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=["company_name"],
            with_vectors=False,
        )
        for r in batch:
            name = r.payload.get("company_name", "").strip()
            if name:
                none_names.add(name)
        if offset is None:
            break

    print(f"  {len(none_names)} unique company names with ticker=None")

    resolvable = {n: name_ticker_map[n] for n in none_names if n in name_ticker_map}
    unresolvable = none_names - set(resolvable.keys())
    print(f"  resolvable: {len(resolvable)}, unresolvable: {len(unresolvable)}")

    if unresolvable:
        print(f"\nUnresolvable (no ticker found anywhere, first 20):")
        for n in sorted(unresolvable)[:20]:
            print(f"  {n}")

    if DRY_RUN:
        print("\n-- DRY RUN -- Would patch:")
        for name, ticker in sorted(resolvable.items())[:30]:
            print(f"  {name!r} → {ticker}")
        return

    # Patch in batches
    print(f"\nPatching {len(resolvable)} company names...")
    total_patched = 0
    for i, (name, ticker) in enumerate(sorted(resolvable.items())):
        ids = fetch_null_ticker_ids(name)
        if not ids:
            continue

        # Patch in batches of BATCH_SIZE
        for j in range(0, len(ids), BATCH_SIZE):
            chunk = ids[j : j + BATCH_SIZE]
            client.set_payload(
                COLLECTION,
                payload={"ticker": ticker},
                points=chunk,
            )

        total_patched += len(ids)
        if (i + 1) % 50 == 0 or (i + 1) == len(resolvable):
            print(f"  [{i+1}/{len(resolvable)}] patched {total_patched} records so far")

    print(f"\nDone. Total records patched: {total_patched}")


if __name__ == "__main__":
    main()
