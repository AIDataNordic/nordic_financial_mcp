# BACKLOG.md — Åpne oppgaver og prioriteringer

## Alexandria

### List Alexandria på punkpeye/awesome-mcp-servers
Fork `punkpeye/awesome-mcp-servers` under AIDataNordic, legg til entry under `### 🎨 Art & Culture`, og lag PR. Entry:
```
- [AIDataNordic/alexandria-mcp](https://github.com/AIDataNordic/alexandria-mcp) [![AIDataNordic/alexandria-mcp MCP server](https://glama.ai/mcp/servers/AIDataNordic/alexandria-mcp/badges/score.svg)](https://glama.ai/mcp/servers/AIDataNordic/alexandria-mcp) 🐍 ☁️ - Semantic search over 4.6M+ text chunks from 20,000+ classical philosophy and humanities works — Plato, Aristotle, Kant, Nietzsche, and thousands more. Multilingual (Ancient Greek, Latin, German, French, English). No install needed.
```
Krever gyldig GitHub-token (nåværende er utløpt).

## Nostr / DVM

### Orkestrator-agent — dynamisk DVM-markedsplass via MCP

Et eget produkt: en MCP-server som eksponerer NIP-90-markedet som verktøy til Claude og andre agenter. Agenten trenger ikke kjenne til Nostr eller Lightning — orkestratoren håndterer alt under panseret.

**Hva den gjør:**
- Abonnerer på kind 31990 fra relayer → bygger dynamisk katalog over tilgjengelige DVM-er (kapabilitet, pris, latens)
- Eksponerer verktøy som `find_dvm_service(task_type)` og `execute_dvm_job(query, budget)`
- Velger billigste/raskeste DVM for oppgaven automatisk
- Betaler bolt11 via LND uten menneskelig godkjenning
- Returnerer kind 6300-resultat til MCP-klienten

**Arkitektur:**
```
Claude-agent (MCP-klient)
        ↓ MCP
Orkestrator-MCP-server  ← dette prosjektet
        ↓ Nostr NIP-90          ↓ Lightning
   DVM-tilbydere (inkl. Alfred)   LND/LNbits
```

**Gjenbrukbart fra eksisterende infrastruktur:**
- LNbits + LND (betaling fungerer allerede)
- `alfred_dvm.py` — referanseimplementasjon av DVM-tilbyder
- `alfred.py` — referanseimplementasjon av MCP-server

**Mangler:**
- [ ] Kind 31990-crawling og dynamisk DVM-katalog
- [ ] DVM-valg basert på pris/latens/kapabilitet
- [ ] Generalisert job-dispatcher (ikke hardkodet til finanssøk)
- [ ] Automatisk Lightning-betaling uten menneskelig godkjenning

**Nostr er discovery-laget** — desentralisert, ingen sentral katalog. Lightning er betalingslaget. MCP er grensesnittet mot agenter. Kombininasjonen er "the machine economy" fra NIP-90-spesifikasjonen i praksis.

### Alfred DVM — demo-case (NIP-90) ✅ (grunnflyt ferdig 2026-05-09)
`alfred_dvm.py` kjører og leverer kind 6300 + kind 1 til Gossip. Se `HANDOFF_NOSTR.md`.

Gjenstående:
- [ ] LND REST API-betaling (bolt11 invoice)
- [ ] `alfred.aidatanorge.no` (Cloudflare Tunnel) for full rapportlenke

### DVM — bytt backend fra Alfred til direkte Qdrant-søk
Alfred (`due_diligence_report`) er for tregt og for bredt for Nostr-formatet (~4 min, full rapport).
Erstatt med direkte semantisk søk via `search_filings` (Nordic Financial MCP, port 8003).

**Tilnærming:**
- Ta company-navn fra kind 5300-request
- Kall `search_filings` med 2-3 målrettede queries (siste nyheter, nøkkeltall, risiko)
- Sett sammen et kort sammendrag (5-8 linjer) av de beste chunks
- Responstid: ~5-10 sekunder i stedet for ~4 minutter

**Filendringer:** `alfred_dvm.py` — erstatt `AlfredClient`-kallet med direkte `search_filings`-kall mot `http://localhost:8003/mcp`.

### DVM — fremtidige tjenester
Tilby følgende DVM-tjenester utover finanssøk:

- **Web scraping** (Playwright, requests) — hent innhold fra URL på oppdrag
- **HTML/PDF-parsing og strukturering** — ekstraher og normaliser dokumenter
- **Chunking og metadata-ekstraksjon** — del opp og berik innhold for videre ingest

Hver tjeneste får egen job-kind og NIP-89-announcement. Kan prises individuelt med LND.

### NIP-90 klient-UX — manglende betalingslag i eksisterende klienter

**Problem:** Ingen av de vanlige Nostr-klientene (Amethyst, Gossip) har UI for å håndtere kind 7000 `payment-required`. Brukeren ser "Job Requested, waiting for reply" og ingenting mer — selv om Alfred sender faktura korrekt.

**Hva som mangler i klientene:**
- Parse `amount`-taggen i kind 7000 (sats-beløp + bolt11-streng)
- Vis betalingsknapp
- Trigger wallet: deep link (`lightning:<bolt11>`) på mobil, eller `window.webln.sendPayment()` i nettleser (krever Alby extension)

**Mulige løsninger, i stigende kompleksitet:**
1. **Bidra til en web-klient** (noStrudel/Coracle/Svelte — TypeScript) — enkleste vei, WebLN fungerer i nettleser
2. **Bidra til Amethyst** (Android/Kotlin, `vitorpamplona/amethyst` på GitHub) — riktig plattform for mobilbrukere, men krever Android Studio og Kotlin-kjennskap
3. **Enkel egen webside** med WebLN — fungerer, men ekstra friksjon for brukeren

**Alby-avklaring:** Alby er en Lightning wallet / browser extension, ikke en fullverdig Nostr-klient. Den gir `window.webln` og `window.nostr` (NIP-07) i nettleseren, men har ikke en mobil-app med NIP-90-støtte.

**Neste steg:** Se på `vitorpamplona/amethyst` sin kodebase — finn der kind 7000 mottas og vurder hva som trengs for å legge til betalings-UI.

### NIP-90 overvåking — løpende
Cron kjører (`nip90_collect.py` + `nip90_report.py`). Daglige rapporter postes til relay.aidatanorge.no og vises i Gossip. Ingen umiddelbare oppgaver, men følg med på:
- Om nye job-kinds dukker opp (finansdata, agent-koordinering)
- Om åpent marked vokser

---

## Høy prioritet

### Alexandria MCP — Glama re-sync
`glama.json` ble endret fra `"type": "streamable-http"` til `"type": "http"` og pushet til GitHub (commit `fd51ee7`). Glama kjører fortsatt serveren lokalt og feiler med OOM på e5-large (2.2GB). **Handling:** Gå til Glama admin-panel og trigger manuell re-sync/rebuild så den nye transport-konfigurasjonen trer i kraft.



### Eval-rammeverk for Alfred — HØYESTE PRIORITET

Siste kjøring: **2026-05-28, 50 selskaper** (`eval_results_20260528.json`)

| Metrikk | Resultat |
|---------|---------|
| Revenue match | 47/50 (94%) |
| news-seksjoner | 0.3/5 snitt — **10% treffrate** |
| financials fritekst | 1.1/5 — 30% treffrate |
| operations/risks | ~1.0/5 — 18-20% treffrate |
| XBRL-seksjoner | 3.6/5 — 82-96% treffrate ✅ |

**Prioriterte forbedringer (i rekkefølge):**

1. **Nyhetshenting er brutt** — undersøk hvorfor `news_YEAR`-seksjoner konsekvent returnerer 0. Sannsynlig rotårsak: søkestreng, kilde-filter eller datofilter i Alfred-probe.
2. **BioGaia ticker-mismatch** — BIOGB i eval, men BGLA i Qdrant. Fiks alias eller probe-logikk.
3. **Konkurrentvalg** — Lagercrantz og Indutrade foreslås for ulike bransjer. Legg til bransje-kontekst i konkurrent-seksjonen eller filtrer generiske industri-selskaper.
4. **Raske kjøringer med feil selskap** — IRLAB (31s), Hampiðjan (37s), AEGA (40s): `ticker_confirmed=None` men perfekt dekning. Trolig treff på annet selskap. Legg til sanity-check: hvis `ticker_confirmed=None`, merk resultatet som usikkert.
5. **Regresjonstesting** — kjør 100-selskaper etter nyhetsfiks for å bekrefte fremgang

### Alfred — neste steg
1. Cloudflare-tunnel + deploy på `alfred.aidatanorge.no`
2. Eval-rammeverk: kjør mot 10–15 selskaper, score seksjondekning
3. Pre-kall til Qdrant for ticker-liste til Haiku (riktig konkurrentidentifisering)
4. Deduplicer chunks på tvers av seksjoner
5. Registrer på Smithery/Glama som eget produkt (repo: `AIDataNordic/alfred-mcp`)

### XBRL-kvalitet — gjennomgang påkrevd ⚠️
Systematisk gjennomgang ikke gjort. Noen chunks inneholder rå XBRL-taggnavn.
1. Kjør kvalitetsskanning — identifiser chunks med høy andel `ifrs-full:`-fragmenter
2. Vurder om ren tekst-ekstraksjon gir bedre resultater enn iXBRL-ekstraksjon
3. Sett terskel: chunks med >X% XBRL-tags forkastes eller re-ingestas
4. Prioriter: Alfa Laval, Volvo, Atlas Copco, ABB

### financial_summary — kvartalstall mangler
`financial_summary_from_xbrl.py` dekker kun årsrapporter (xbrl_esef). Kvartalstall (EBITDA/EBIT fra newsweb/mfn_nordics) er ikke ekstrahert strukturert. Revenue er typisk i tabeller — ikke regex-tilgjengelig. Krever bedre tabellekstraksjon fra HTML-pressemeldinger.

## Medium prioritet

### XBRL ticker-alias mismatch (extracted_xbrl vs. børsticker)
Noen selskaper har LEI-baserte tickers i extracted_xbrl som ikke matcher børstickeren — kjent eksempel: DNQ (Equinor ASA i XBRL) vs. EQNR (Oslo Børs). 523 unike tickers i extracted_xbrl, ukjent antall mismatches.
- Sammenlign extracted_xbrl-tickers mot newsweb/mfn_nordics for samme selskap (via company_name eller LEI)
- Bygg alias-tabell og patch Qdrant-payload med korrekt ticker
- Alternativt: legg til alias-oppslagtabell i `mcp_server.py` ved query-tid

### XBRL ticker=None backfill
Flere selskaper har XBRL-poster i `xbrl_esef` med `ticker=None` — bl.a. ØRSTED A/S, AGF A/S, Neste Oyj. Gjør at Alfred-probe ikke kan bekrefte ticker selv om data finnes.
- Finn alle `xbrl_esef`-records med `ticker=None` via Qdrant scroll
- For hvert unike `company_name`: finn riktig ticker fra andre poster for samme selskap (andre kilder, eller andre XBRL-poster med ticker satt)
- Patch med `client.set_payload()`
- Berørte selskaper inkluderer ØRSTED (DNNGY), AGF, Neste (NTOIY) og sannsynligvis flere

### Nasdaq SE-ingest (pågår)
Kjører per 2026-05-05. Forventes 2–3M vektorer totalt. Sjekk status:
```bash
tail -f ~/logs/nasdaq_se_ingest.log
```

### Prompt-forbedringer
- `company_analysis`-prompten bør nevne strømpris for kraftintensive selskaper (aluminium, stål, oppdrett, datasentre). Legg til: "Hvis selskapet er kraftintensivt, hent strømpris med `get_current_power_price`."

### `get_company_info` — mangler SE og IS
Live Brønnøysund-lookup støtter NO, DK, FI. SE (Bolagsverket) og IS (Fyrirtækjaskrá) mangler. Begge har gratis APIer.

### Makrodata for SE/DK/FI er ufullstendig
`macro_ingest_nordic.py` mangler: byggekostnadsindekser, handelsindekser, kredittvekst, boligpriser for SE (SCB), DK (Danmarks Statistik), FI (Statistics Finland). Alle har gratis PX-Web/JSON-APIer. NO-implementasjonen i `macro_ingest.py` er malen.

### Cron mangler for kvartalsdata
- `commodity_ingest.py` — kvartalssummeringer. Forslag: `0 5 1 1,4,7,10 *`
- `macro_ingest.py` / `macro_ingest_nordic.py` — samme mønster

### ENTSO-E utvidelse
Nåværende ingest: kun day-ahead priser (A44).
- **A79 — Vannmagasinfylling** — sterkeste enkeltindikator for NO-priser. `get_hydro_reservoir_level()`
- **A75 — Produksjonsmiks** — grunnlag for karbonintensitet. `get_carbon_intensity()`
- **A71 — Vind/solprognoser** — prisprediksjonsgrunnlag
- Referanse: [kraftsystemet/awesome-kraftsystemet](https://github.com/kraftsystemet/awesome-kraftsystemet)

### x402 micropayments — infrastruktur klar, deaktivert
Kommentert ut i `mcp_server.py` inntil mainnet. Aktivering: fjern `#`-ene i x402-blokken, legg til `_meta: dict = None` i `search_filings` og `parse_pdf_to_text`.
- Planlagte priser: `search_filings` $0.05, `parse_pdf_to_text` $0.02, `due_diligence_report` ~$1–2
- EVM: `0x1C903401F5725a0aA839fA0321b183E0A488D3c6` (Base mainnet: `eip155:8453`)
- `mcp_server_x402.py` og `nordic-mcp-x402.service` kan pensjoneres når x402 er aktivt

### analyze_company i mcp_server.py
Implementert men ikke testet (Anthropic API-nøkkel var ikke aktiv). Verifiser at nøkkelen virker. Test: `analyze_company("Mowi", "Hva var driftsinntektene i 2023?")`

### Nye MCP-verktøy
- `summarize_company` — 15 chunks bredt på tvers av `fiscal_year`. Pris ~$0.10–0.25
- `due_diligence_report` i `mcp_server.py` — strukturerte chunks per tema. Pris ~$1–5

## Lav prioritet / fullført historikk

### Registrering på registre
- **Smithery** ✅ — live på `https://smithery.ai/servers/kontakt-qy0g/nordic-financial-mcp`. Publisering: `smithery mcp publish "https://mcp.aidatanorge.no/mcp" -n "kontakt-qy0g/nordic-financial-mcp" --config-schema /tmp/cs.json`
- **MCP Registry** ✅ — live
- **Glama** ✅ — live. TODO: ny build etter awesome-mcp-servers-godkjenning. Build steps: se git-historikk.
- **awesome-mcp-servers** — Food Recipe MCP PR #5555 åpen. Nordic Financial MCP PR åpen, venter på godkjenning (2026-05-14).

### Alexandria MCP-server ✅
- Live: `https://alexandria.aidatanorge.no/mcp`
- `philosophy_russia.py` — kjøres på Vast.ai (2026-05-25), ~6 000 russiske tekster ✅ (pågår)
- TODO: **Cleanup feilmerkede russiske tekster** — Archive.org sin `language:russian`-metadata er upålitelig for `bub_gb_`-identifikatorer (Google Books/europeiske biblioteker). Etter fullført ingest: kjør cleanup-script som sletter Alexandria-chunks der `language=rus` men `ocr_detected_lang` ikke er russisk. Lag `philosophy_russia_cleanup.py` med dry-run først.

### XBRL historiske fikser ✅
- Re-ingest av 752 korrupte filings (2026-04-21)
- Coverage-fix: `xbrl_coverage_fix.py` — kan gjenbrukes ved fremtidige coverage-problemer
- `?`-kilde bulk-oppdatert til `source="xbrl_esef"` (10 352 vektorer)
- XBRL_FACT_TAGS utvidet (2026-05-05): `RevenueFromContractsWithCustomers` + balanse-underposter
- Scale-normalisering lagt til i `extract_xbrl_facts()` (2026-05-05)
- `xbrl_revenue_patch.py` kjørt: patcher revenue_eur for ~1144 docs via re-nedlasting av zip

### financial_summary-chunks ✅ (2026-05-05)
`financial_summary_from_xbrl.py` upserted 936 chunks (`report_type="financial_summary"`, `source="extracted_xbrl"`) med strukturerte balanse- og revenue-tall fra eksisterende XBRL-payload. Én chunk per (ticker, fiscal_year). Balanse: 99% dekning. Revenue: 66% + patch.
