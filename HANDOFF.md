# Session Handoff — 2026-05-08 (oppdatert)

## System State
- `mcp_server.py` port 8003, FAST_MODE=1 (via systemd override) — kjører
- `alfred.py` port 8006 (PID 3353657) — kjører med ny probe-kode
- Qdrant localhost:6333, ~1.109M vektorer — grønn

## Siste commits
- `37a9b82` — Probe ticker-resolusjon: XBRL-prioritet, xbrl_esef navne-fallback (NOD→NRSDY, GCC→GCCNOK, CapMan)
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

## eval_results_20260508.json — 62/100
Kjørt 2026-05-08. Gjenstående feilkategorier:
1. **ticker=None i XBRL** (~3 selskaper bekreftet: ØRSTED/DNNGY, AGF, Neste/NTOIY) — se BACKLOG
2. **Datahull** — PHO, DSV og flere mangler XBRL helt
3. **Svensk ingest ufullstendig** — ASSAB, CCC, etc. mangler XBRL inntil SE-ingest er ferdig
4. **fiscal_year-gap** — KONE/KNYJY har XBRL kun til FY2023, Alfred søker FY2025

## Gjenstående problemer
| Problem | Status |
|---------|--------|
| XBRL ticker=None backfill | Åpent — se BACKLOG |
| Nasdaq SE-ingest (Vast) | Stoppet pga. Qdrant-timeout på Vast — restart når Qdrant er oppe |
| Re-kjøre eval etter SE-ingest | Vent til SE-ingest er ferdig for representativt tall |

## Nøkkelfiler
- `eval_results_100.json` — 80/100 baseline (gyldig, mai 2026)
- `eval_results_100_v3.json` — 42/100, ikke representativ (se over)
- `xbrl_scale_suspects.json` — 205 hard / 176 review suspects
- `/etc/systemd/system/nordic-mcp.service.d/override.conf` — FAST_MODE=1
