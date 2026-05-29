#!/usr/bin/env python3
"""
newsweb_ingest2.py — NewsWeb → Qdrant

Bruk:
  python3 newsweb_ingest2.py --backfill --start-id 640000
  python3 newsweb_ingest2.py --backfill --start-id 500000 --end-id 669999
  python3 newsweb_ingest2.py --backfill --start-id 669980 --dry-run
  python3 newsweb_ingest2.py --update

Cron:
  0 7 * * * /home/mini-hal/norsk-mcp-server/venv/bin/python3 \
            /home/mini-hal/norsk-mcp-server/newsweb_ingest2.py --update \
            >> /home/mini-hal/logs/ingest_newsweb.log 2>&1
"""

import sys, time, json, hashlib, re, argparse, requests
from pathlib import Path
from datetime import datetime, timezone

API_BASE     = "https://api3.oslo.oslobors.no/v1/newsreader"
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "nordic_company_data"
DENSE_MODEL  = "intfloat/e5-large-v2"
SPARSE_MODEL = "Qdrant/bm25"
STATE_FILE   = Path(__file__).parent / "newsweb_state.json"
LOG_FILE     = Path(__file__).parent / "logs" / "newsweb_ingest.log"
SOURCE       = "newsweb"
WANTED       = {1001: "annual_report", 1002: "quarterly_report", 1104: "press_release"}
BATCH_SIZE   = 50
DELAY        = 0.15

# NewsWeb market-streng → ISO land-kode
MARKET_TO_COUNTRY = {
    "Oslo Børs":           "NO",
    "Oslo Axess":          "NO",
    "Euronext Growth Oslo": "NO",
    "Merkur Market":       "NO",
    "Nasdaq Stockholm":    "SE",
    "Nasdaq Helsinki":     "FI",
    "Nasdaq Copenhagen":   "DK",
    "Nasdaq First North":  "SE",   # default SE — kan overstyres ved behov
}


# === HJELPEFUNKSJONER ===

def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build_point_id(doc_id: str, chunk_index: int) -> int:
    """Deterministisk, kollisjonssikker uint64 for Qdrant."""
    raw = sha256_hex(f"{doc_id}|{chunk_index}")
    return int(raw[:16], 16) % (2 ** 63)


def extract_fiscal_period(title: str, published: str) -> tuple:
    """
    Utleder regnskapsår (int) og kvartal (int|None) fra tittel og publiseringsdato.
    """
    year_match = re.search(r"(20\d{2})", title)
    year = int(year_match.group(1)) if year_match else None
    if year is None and published:
        try:
            year = int(published[:4])
        except (ValueError, IndexError):
            pass

    t = title.lower()
    quarter = None
    if any(x in t for x in ["q1", "first quarter", "jan-mar", "three months ended mar",
                              "january to march", "1st quarter"]):
        quarter = 1
    elif any(x in t for x in ["q2", "second quarter", "half-year", "half year",
                                "six months", "jan-jun", "h1 "]):
        quarter = 2
    elif any(x in t for x in ["q3", "third quarter", "nine months", "jan-sep",
                                "january to september"]):
        quarter = 3
    elif any(x in t for x in ["q4", "fourth quarter", "full year", "full-year",
                                "year-end", "annual", "bokslut"]):
        quarter = 4

    return year, quarter


def infer_country(msg: dict) -> str:
    """Utleder landkode fra market-feltet i NewsWeb-meldingen."""
    markets = msg.get("markets") or []
    for m in markets:
        country = MARKET_TO_COUNTRY.get(m)
        if country:
            return country
    return "NO"   # NewsWeb er Oslo Børs-basert — NO er trygt default


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_ingested_id": 0, "total_ingested": 0, "last_run": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_max_id():
    try:
        r = requests.post(f"{API_BASE}/list", json={}, timeout=(10, 15))
        msgs = r.json().get("data", {}).get("messages", [])
        if msgs:
            return max(m["messageId"] for m in msgs)
    except Exception as e:
        log(f"  Advarsel: unntak under henting av max messageId: {e}")
    return 0


def fetch_message(mid):
    r = requests.get(f"{API_BASE}/message", params={"messageId": mid}, timeout=(10, 15))
    if r.status_code == 200:
        return r.json().get("data", {}).get("message")
    return None


def build_text(msg):
    cat = (msg.get("category") or [{}])[0]
    return "\n".join([
        f"Selskap: {msg.get('issuerName', '')} ({msg.get('issuerSign', '')})",
        f"Dato: {msg.get('publishedTime', '')[:10]}",
        f"Type: {cat.get('category_en', '')}",
        f"Tittel: {msg.get('title', '')}",
        "",
        msg.get("body", ""),
    ]).strip()


def chunk_text(text, max_chars=1500, overlap=200):
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut == -1:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return [c for c in chunks if len(c) > 50]


def upsert_batch(points):
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points},
        timeout=(10, 60),
    )
    if r.status_code not in (200, 206):
        log(f"  FEIL Qdrant: {r.status_code} {r.text[:100]}")


# === INGEST ===

def run(start_id, end_id, dry_run):
    dense_model = None
    sparse_model = None
    if not dry_run:
        log("Laster embedding-modeller...")
        from sentence_transformers import SentenceTransformer
        from fastembed import SparseTextEmbedding
        dense_model  = SentenceTransformer(DENSE_MODEL, device="cpu")
        sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        log("Modeller lastet.")

    total, total_vectors, highest, batch = 0, 0, start_id - 1, []
    log(f"Starter: {start_id} → {end_id} (~{end_id-start_id:,} meldinger)")

    for mid in range(start_id, end_id + 1):
        time.sleep(DELAY)
        try:
            msg = fetch_message(mid)
        except Exception as e:
            log(f"  FEIL {mid}: {e}")
            continue

        if not msg:
            continue

        highest = max(highest, mid)
        cat_id = (msg.get("category") or [{}])[0].get("id")
        if cat_id not in WANTED:
            continue

        report_type = WANTED[cat_id]
        text = build_text(msg)
        if len(text) < 100:
            continue

        chunks      = chunk_text(text)
        published   = msg.get("publishedTime", "")
        title       = msg.get("title", "")
        country     = infer_country(msg)
        markets     = msg.get("markets") or []
        market_str  = markets[0] if markets else ""
        fiscal_year, fiscal_quarter = extract_fiscal_period(title, published)
        doc_id      = sha256_hex(f"{SOURCE}|{mid}")[:32]
        now_iso     = datetime.now(timezone.utc).isoformat()

        if dry_run:
            log(f"  [DRY] {mid} | {msg.get('issuerSign','?'):8} | {country} | {report_type:15} | {title[:50]}")
            total += 1
            continue

        # Dense embeddings — passage:-prefiks for e5-large-v2
        prefixed = [f"passage: {c}" for c in chunks]
        dense_embeddings = dense_model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        )

        # Sparse embeddings — BM25 via fastembed (uten prefiks)
        sparse_results = list(sparse_model.embed(chunks))

        for i, (chunk, dense_emb, sparse_res) in enumerate(
            zip(chunks, dense_embeddings, sparse_results)
        ):
            point_id = build_point_id(doc_id, i)
            batch.append({
                "id": point_id,
                "vectors": {
                    "dense":  dense_emb.tolist(),
                    "sparse": {
                        "indices": sparse_res.indices.tolist(),
                        "values":  sparse_res.values.tolist(),
                    },
                },
                "payload": {
                    # Identifikasjon
                    "source":         SOURCE,
                    "doc_id":         doc_id,
                    "url":            f"https://newsweb.oslobors.no/message/{mid}",

                    # Selskap
                    "company_name":   msg.get("issuerName", ""),
                    "ticker":         msg.get("issuerSign", ""),
                    "isin":           msg.get("isin") or None,
                    "market":         market_str,
                    "country":        country,

                    # Dokument
                    "title":          title,
                    "language":       "no",
                    "published_date": published,
                    "report_type":    report_type,
                    "fiscal_year":    fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "text_source":    "api",
                    "message_id":     msg.get("messageId"),

                    # Chunk
                    "chunk_index":    i,
                    "total_chunks":   len(chunks),
                    "text":           chunk,
                    "ingested_at":    now_iso,
                },
            })
        total += 1
        total_vectors += len(chunks)

        if len(batch) >= BATCH_SIZE:
            upsert_batch(batch)
            batch = []
            log(f"  Fremgang: id={mid}, ingested={total}")

    if batch and not dry_run:
        upsert_batch(batch)

    return total, total_vectors, highest


# === MAIN ===

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true")
    mode.add_argument("--update",   action="store_true")
    parser.add_argument("--start-id", type=int, default=640000)
    parser.add_argument("--end-id",   type=int, default=0)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    state = load_state()

    log("Henter høyeste messageId...")
    max_id = get_max_id()

    if not max_id:
        fallback = state.get("last_ingested_id", 0)
        if fallback and args.update:
            log(f"API returnerte tom liste — ingen nye meldinger siden sist (siste kjente id={fallback}). Avslutter.")
            return
        elif args.end_id:
            log(f"API utilgjengelig, bruker manuelt angitt --end-id={args.end_id}")
            max_id = args.end_id
        else:
            log("FEIL: API utilgjengelig. Bruk --end-id 669999 for å angi sluttpunkt manuelt.")
            sys.exit(1)
    else:
        log(f"Høyeste messageId: {max_id}")

    if args.backfill:
        start_id, end_id = args.start_id, max_id
    else:
        start_id = state.get("last_ingested_id", 0) + 1 or max_id - 500
        end_id   = max_id
        if start_id > end_id:
            log("Ingen nye meldinger.")
            return

    t0 = time.time()
    ingested, total_vectors, highest = run(start_id, end_id, dry_run=args.dry_run)

    if not args.dry_run and highest > state.get("last_ingested_id", 0):
        state.update({
            "last_ingested_id": highest,
            "total_ingested":   state.get("total_ingested", 0) + ingested,
            "last_run":         datetime.now(timezone.utc).isoformat(),
        })
        save_state(state)

    log(f"Ferdig! ingested={ingested}, nye vektorer={total_vectors}, høyeste_id={highest}, tid={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
