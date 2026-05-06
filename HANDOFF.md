# Session Handoff — Alfred eval, 2026-05-06

## System State
- `mcp_server.py` port 8003, `alfred.py` port 8006 — begge kjører
- Qdrant localhost:6333, collection `nordic_company_data`
- 1519 financial_summary chunks (source=extracted_xbrl), 600 selskaper

## Ucommittede endringer (git)
- `alfred.py`: Finnish probe-fallback i `_probe()` — søker xbrl_esef når nasdaq_fi-chunks mangler ticker
- `eval_alfred.py`: Nytt eval-rammeverk (ikke tracked av git)
- `financial_summary_from_xbrl.py`: Fjernet chunk_index=0-filter (kjørt, ikke committet)

## Eval-status
- Baseline 100-company eval: 80/100 (eval_results_100.json)
- Etter Finnish probe-fix: ~93/100 (ikke re-kjørt, basert på targeted rerun av de 20 som feilet)
- ALVO financial_summary regenerert (revenue manglet fra teksten)

## Gjenstående problemer
| Ticker | Problem | Fix |
|--------|---------|-----|
| ETTE | GT scale-bug (359 951M EUR, bør være ~360M) | Fikset: lagt til BAD_GROUND_TRUTH |
| KEMPOWR | GT scale-bug (223 697M EUR) | Fikset: lagt til BAD_GROUND_TRUTH |
| PON1V | Dry-run av financial_summary_from_xbrl.py fant bare 1 KPI-kandidat og den eksisterer allerede; nyere XBRL-chunks har ikke nok strukturerte KPI-felt til summary | Undersøk XBRL-fact extraction / rapportstruktur for 2023–2024 |
| BETCO | Eval finner XBRL, men revenue parser feil tall (119 vs 371,487 DKK) | Undersøk revenue parsing / XBRL chunk ordering |
| TGS | Targeted eval OK etter rerun | Ingen kjent fix nødvendig nå |
| STRO | XBRL-data finnes, men én targeted eval-run returnerte 0 company-filtered chunks; isolert Alfred-kall returnerte XBRL normalt | Bruk ny Alfred section-logging hvis problemet reproduseres |
| NOTE | Finner ~50% av GT (ticker-konflikt?) | Undersøk NOTE-tickers i Qdrant |

## Neste steg (prioritert)
1. Undersøk XBRL-fact extraction / rapportstruktur for PON1V 2023–2024
2. Kjør full 100-company eval på nytt: `venv/bin/python3 eval_alfred.py --auto 100 --output-json eval_results_100_v2.json`
3. Undersøk NOTE-ticker-konflikt
4. Commit selektivt: `git add alfred.py eval_alfred.py financial_summary_from_xbrl.py HANDOFF.md Agents.md`

## Nøkkelfiler
- `alfred.py` — due diligence orchestrator (port 8006)
- `mcp_server.py` — Nordic Financial MCP (port 8003)
- `eval_alfred.py` — eval-rammeverk
- `financial_summary_from_xbrl.py` — bygger financial_summary chunks fra XBRL
- `eval_results_100.json` — 100-company baseline
- `eval_rerun_failures.json` — targeted rerun av de 20 som feilet
