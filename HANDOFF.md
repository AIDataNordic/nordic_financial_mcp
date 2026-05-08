# Session Handoff — 2026-05-08 (oppdatert)

## System State
- `mcp_server.py` port 8003, FAST_MODE=1 (via systemd override) — kjører
- `alfred.py` port 8006 (PID 3353657) — kjører med ny probe-kode
- Qdrant localhost:6333, ~1.109M vektorer — grønn

## Siste commits
- `2121745` — Probe ticker-validering, eval FY/chunk-rekkefølge, FAST_MODE

## eval_results_100_v3.json — 42/100 (ikke representativ)
Kjørt 2026-05-08 med FAST_MODE. Lavt score skyldes:
1. **Qdrant-transienter** — ~35 selskaper fikk xbrl=0 pga. midlertidig tom respons under lang kjøring
2. **Ticker-mismatch** — 14 selskaper fikk feil confirmed_ticker (nå fikset i probe)
3. **ADR vs. børsticker** — NRSDY/NOD, ASAZY/ASSAB, ILKKA2/ILK1S (fikset — se under)

## ADR-ticker fix (2026-05-08)
Rotårsak: `_probe()` lot `verified_input_ticker` (hentet fra brukerinput eller press release-rader) overstyre XBRL-tickeren. For ADR-selskaper finnes XBRL-data under primærbørs-ticker (f.eks. "NHY"), ikke ADR-ticker ("NRSDY"), som ga 0 XBRL-treff.

Fix i `alfred.py` — tre endringer i `_probe()`:
1. `general_rows` bevares separat fra `rows` (som kan overskrives av ticker-filtrert søk)
2. XBRL-tickers ekstraheres alltid fra `general_rows` (ufiltrert)
3. Prioritetsrekkefølge snudd: XBRL-ticker → verified_input_ticker → all_tickers

Ingen hardkodet alias-tabell — løsningen er selvkorrigerende for alle fremtidige ADR-/alias-tilfeller.

## Probe-forbedringer (committed)
- GUBRA: SPG → GUBRA ✓
- ORIGO: SKAB → ORIGO ✓
- Gram Car Carriers: GCC → GCCNOK ✓
- ARR: FAG → ARR ✓
- 5 andre: returnerer nå None i stedet for feil selskap

## FAST_MODE
`FAST_MODE=1` er satt i `/etc/systemd/system/nordic-mcp.service.d/override.conf`.
Skrur av cross-encoder reranking — reduserer latens fra ~250s til ~70s per selskap på CPU.
For produksjon bør dette vurderes fjernet (reranking gir bedre kvalitet).
Skru av: `sudo rm /etc/systemd/system/nordic-mcp.service.d/override.conf && sudo systemctl daemon-reload && sudo systemctl restart nordic-mcp.service`

## Gjenstående problemer
| Problem | Status |
|---------|--------|
| ADR-ticker aliasing (NRSDY/NOD, etc.) | Fikset — XBRL-ticker prioriteres over verified_input_ticker |
| Island-ingest krasjet (Qdrant-timeout) | Kan restartes: `nohup venv/bin/python3 nasdaq_is_ingest.py >> ~/logs/nasdaq_is_ingest.log 2>&1 &` (hopper over prosesserte via `nasdaq_is_processed.txt`) |
| Batch-patch hard suspects (205 stk) | Kjørt — bekreftet via Qdrant (f.eks. TRH1V FY2022: 975336840 → 975.34 EURm) |
| Re-kjøre eval uten Qdrant-transienter | Neste klare eval bør kjøres på et rolig tidspunkt |

## Nøkkelfiler
- `eval_results_100.json` — 80/100 baseline (gyldig, mai 2026)
- `eval_results_100_v3.json` — 42/100, ikke representativ (se over)
- `xbrl_scale_suspects.json` — 205 hard / 176 review suspects
- `/etc/systemd/system/nordic-mcp.service.d/override.conf` — FAST_MODE=1
