# Session Handoff — 2026-05-29 (oppdatert)

## System State
- `mcp_server.py` port 8003, FAST_MODE=0 — kjører
- `alfred.py` port 8006 — kjører
- `alfred_dvm.py` — kjører som **systemd service** (`alfred-dvm.service`)
- `lnbits` port 5000 — kjører (tmux session `lnbits`)
- Qdrant localhost:6333, ~1.78M vektorer — grønn
- nostream relay — kjører, `wss://relay.aidatanorge.no`

## Alfred DVM — oppdatert arkitektur (2026-05-09 kveld)

**Start/restart:** `sudo systemctl restart alfred-dvm`
**Logg:** `tail -f ~/logs/alfred_dvm.log`
**DVM-nøkkel:** `npub1ncgh88pe8gq6uj8gve45y04pyrhn6tfw8pk8mm8kynqzk4tc3mks5w03hl`
**Nøkkelfil:** `dvm_keys.json`

### Backend-bytte — FERDIG ✅
DVM kaller nå `search_filings` direkte på Nordic Financial MCP (port 8003) — ikke Alfred lenger.
- Responstid: ~15-20 sek (ned fra ~4 min)
- 3 sekvensielle kall per job: ticker-resolusjon → `extracted_xbrl` financial summary → nyhetssøk
- `fast=True` per-kall parameter lagt til i `mcp_server.py` (hopper over reranking for DVM-kall)

### Støttede input-kanaler
- **kind 5300** — NIP-90 job request (åpent marked)
- **kind 4** — NIP-04 kryptert DM til DVM-pubkey (svar som kind 4 DM)
- **kind 1 mention** — `@alfred <selskapsnavn>` i vanlig note (svar som kind 1 reply)

### Profil og relay-konfig
- DVM publiserer kind 0 profil med navn "alfred" ved oppstart
- DVM publiserer kind 10050 (DM inbox relay: `relay.aidatanorge.no`) ved oppstart
- Relay NIP-11 info oppdatert i `/opt/nostream/.nostr/settings.yaml`:
  - name: "AI Data Nordic Relay"
  - contact: "kontakt@aidatanorge.no"
  - pubkey: brukerens hex-pubkey (fra npub1j5dunrpe...)
  - maxSubscriptions: 25 (opp fra 10)

### Kjente begrensninger
- `extracted_xbrl`-finansdata bruker LEI-baserte tickers (f.eks. DNQ for Equinor EQNR) — se BACKLOG.md
- Ingen selskapsbeskrivelse i output ennå
- LND-betaling ikke implementert ennå

## LNbits — installert 2026-05-10 ✅

LNbits 1.5.4 kjører på mini-hal (port 5000), koblet til LND på Raspiblitz via Tailscale.
- **Start:** `tmux new-session -d -s lnbits "cd /home/mini-hal/lnbits-app && LNBITS_DATA_FOLDER=/home/mini-hal/lnbits/data /home/mini-hal/lnbits-venv/bin/lnbits --host 0.0.0.0 --port 5000 2>&1 | tee -a ~/logs/lnbits.log"`
- **Logg:** `tail -f ~/logs/lnbits.log`
- **Web-UI:** SSH-tunnel fra laptop: `ssh -L 5000:localhost:5000 mini-hal@100.125.117.115` → `http://localhost:5000`
- **Detaljer:** Se `HANDOFF_LND.md`

**Neste steg:**
- Implementere LNbits-betaling i `alfred_dvm.py` (`POST /api/v1/payments`)
- `alfred.aidatanorge.no` Cloudflare Tunnel

## Alexandria MCP — Glama-oppsett fullført 2026-05-12

Repo: `AIDataNordic/Alexandria-mcp` (merk: capital A i GitHub-URL)

Følgende ble gjort for å oppnå full Glama-score:

1. **`glama.json`** — lagt til med `name`, `description`, `categories`, `tags`, `examples` og `transport.streamable-http`
2. **`mcp.json`** — lagt til (samme mønster som Nordic Financial)
3. **`MCP_TRANSPORT` env-støtte** i `alexandria_mcp_server.py` — `os.getenv("MCP_TRANSPORT", "http")`, med stdio-branch for Glama sin `mcp-proxy`-wrapper
4. **Glama Build-konfig** (fylles inn i Glama admin, ikke i repo):
   - Base image: `debian:bookworm-slim`, Node 25, Python 3.11
   - Build steps: curl ned server + requirements, sed-patch transport, uv install, forhåndslast alle tre ML-modeller
   - CMD arguments: `["mcp-proxy", "--", "/app/.venv/bin/python", "/app/alexandria_mcp_server.py"]`
   - JSON schema: `{"properties": {"MCP_TRANSPORT": {"type": "string"}}, "required": [], "type": "object"}`
   - Modell: `intfloat/multilingual-e5-large` (ikke `e5-large-v2` som Nordic Financial)
5. **CI-workflow** — `.github/workflows/ci.yml`, syntax-sjekk av `alexandria_mcp_server.py`
6. **GitHub release `v1.0.0`** — opprettet manuelt på GitHub

**Status:** Build + Publish kjørt. Maintenance/Server Coherence kan ta noen timer å oppdatere.

## Siste commits
- `37a9b82` — Probe ticker-resolusjon: XBRL-prioritet, xbrl_esef navne-fallback
- `2121745` — Probe ticker-validering, eval FY/chunk-rekkefølge, FAST_MODE

## Datakvalitetsforbedringer gjort 2026-05-08 (kveld)

### 1. ticker=None backfill — FERDIG
`xbrl_ticker_patch.py` patchet 10 560 xbrl_esef-records med korrekt ticker via company-name-matching.
- Alle 12 eval-selskaper med "har XBRL men revenue=None" er nå patchet:
  ØRSTED/DNNGY, AGF/AGFB, Neste/NTOIY, AGILLIC, CHEMOMETEC, Solteq, CapMan, Reka, Biohit, Brim, Ericsson, Componenta
- 16 519 records gjenstår med ticker=None (357 unike LEIs — genuint ikke-børsnoterte)

### 2. GLEIF/OpenFIGI ticker-lookup — FERDIG
`gleif_ticker_patch.py` kjørt mot de 357 gjenværende LEIs.
- 21 tickers resolvert, 165 records patchet (LATOB/56, INTEAB/29, COLOB/10, WALLB/4, m.fl.)
- 224 av 357 har ingen ISIN i GLEIF (obligasjonsutstedere, holdingselskaper — forventes)
- RTX hoppet over (Raytheon, ikke-nordisk)

### 3. Scale-bug i revenue_eur — PÅGÅR (bakgrunn)
Rotårsak: records ingestert pre-2026-05-05 hadde `revenue_eur` satt (men feil, uten scale-normalisering).
`xbrl_revenue_patch.py` hoppet over disse siden verdien ikke var NULL.

`xbrl_reparse_patch.py` re-laster og re-parser XBRL-filer med korrekt kode:
- Hard suspects (96 tickers): `nohup venv/bin/python3 xbrl_reparse_patch.py --from-json xbrl_scale_suspects.json --severity hard --apply >> ~/logs/xbrl_reparse.log 2>&1 &`
  - **203/205 var allerede fikset** — bare 2 gjenstod
- Review suspects (inkl. GYLDB FY2023: 720405→720.405): `nohup venv/bin/python3 xbrl_reparse_patch.py --from-json xbrl_scale_suspects.json --severity review --apply >> ~/logs/xbrl_reparse_review.log 2>&1 &`
  - Kjørte ~11 fixes med ratio=1000 ved siste sjekk — fortsatt i gang

Sjekk fremdrift:
```
grep -c "ratio old/new=1000" ~/logs/xbrl_reparse_review.log && grep "Done\." ~/logs/xbrl_reparse_review.log
```

## Eval-resultater — 2026-05-28 (50 selskaper)

Fil: `eval_results_20260528.json`

| Metrikk | Resultat |
|---------|---------|
| Revenue match | **47/50 (94%)** |
| Ticker ikke bekreftet | 3 (Hampiðjan, AEGA, IRLAB) |
| Gjennomsnittlig kjøretid | 134s (min 31s, max 205s) |

### Seksjondekning (snitt 0–5 / treffrate)
| Seksjon | Snitt | Treff |
|---------|-------|-------|
| xbrl_summary | 3.6 | 96% |
| xbrl_financials | 3.7 | 82% |
| xbrl_risks | 2.7 | 94% |
| competitors | 2.8 | 93% |
| macro/power | 2.8 | 100% |
| financials (fritekst) | 1.1 | 30% |
| operations | 1.0 | 20% |
| risks (fritekst) | 0.9 | 18% |
| **news** | **0.3** | **10%** |

### Funn og kjente svakheter

**Nyhetshenting er brutt** — 10% treffrate på tvers av alle 114 nyhetsspørsmål. Den klart største forbedringmuligheten.

**Feil selskap ved raske kjøringer** — IRLAB (31s), Hampiðjan (37s) og AEGA (40s) fullførte med perfekt seksjondekning men `ticker_confirmed=None`. Trolig treff på feil selskap — se som data-støy, ikke suksess.

**Konkurrentvalg er LLM-gjetting** — Lagercrantz og Indutrade dukker opp som "konkurrenter" for svært ulike selskaper (fotballklubb, MarTech, plastprodukter). Seksjonene scorer 3 (relé-nivå, alltid funnet) men er meningsløse for mange selskaper.

**Revenue-feil:**
- `United Bankers Oyj` — finansielt selskap, revenue=None (mangler riktig XBRL-tag?)
- `Heimar hf.` — islandsk eiendom, revenue=None
- `BioGaia AB` — ticker BIOGB mismatch (bekreftet som BGLA), 0 XBRL-chunks

### Neste forbedringssteg
1. **Nyhetshenting** — undersøk hvorfor news-seksjoner konsekvent returnerer 0
2. **BioGaia ticker-mismatch** — BIOGB vs BGLA
3. **Konkurrentfiltrering** — ikke foreslå Lagercrantz/Indutrade for ikke-industriselskaper

## Scale-bug — rotårsak og full forklaring
iXBRL: `scale=3, format="ixt:num-comma-decimal", value="720.405"` → `parse_numeric` tolker `.` som europeisk tusenskilletegn → returnerer 720405. Uten scale-normalisering (pre-2026-05-05-ingest) lagres 720405 million DKK (1000x for høyt). Med korrekt parser: 720405 × 10³ / 10⁶ = 720.405M ✓.

Ny patch trengs hvis nye selskaper ingesteres med scale-bug. Kjør da:
```
venv/bin/python3 xbrl_scale_diagnostic.py --json xbrl_scale_suspects_new.json
venv/bin/python3 xbrl_reparse_patch.py --from-json xbrl_scale_suspects_new.json --severity hard --apply
```

## Gjenstående problemer
| Problem | Status |
|---------|--------|
| Scale-reparse review suspects | Kjører i bakgrunn |
| Re-kjøre eval | Klar når reparse er ferdig |
| Nasdaq SE-ingest | Stoppet pga. Qdrant-timeout — restart ved behov |
| XBRL ticker=None (357 LEIs) | Uløselig — genuint ikke-børsnoterte enheter |

## Nøkkelfiler
- `eval_results_20260508.json` — 62/100 (siste gyldige, pre-datakvalitetsfiks)
- `xbrl_ticker_patch.py` — company-name-basert ticker backfill
- `gleif_ticker_patch.py` — GLEIF/OpenFIGI ticker-lookup (nytt)
- `xbrl_reparse_patch.py` — scale-bug fix via re-parsing
- `xbrl_scale_suspects.json` — 381 suspects (203 hard allerede fikset, 176 review pågår)
- `/etc/systemd/system/nordic-mcp.service.d/override.conf` — FAST_MODE=1
