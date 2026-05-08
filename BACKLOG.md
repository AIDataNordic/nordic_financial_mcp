# BACKLOG.md — Åpne oppgaver og prioriteringer

## Høy prioritet

### Eval-rammeverk for analyze_company / Alfred — HØYESTE PRIORITET
`test_scores.py` er en spire. Mål: systematisk iterasjon på søkestrategi og system prompt.
1. **Golden queries** — kuratert sett med fasitsvar: "Mowi driftsinntekter 2023 = EUR 5,5 mrd"
2. **Automatisk scoring** — kjør mot golden queries, evaluer (eksakt match eller LLM-as-judge)
3. **Iterasjon** — bruk eval til å forbedre søkestrategi, ticker-bruk, manglende data-håndtering
4. **Regresjonstesting** — sikre at forbedringer ikke ødelegger eksisterende queries

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
- **awesome-mcp-servers** — Food Recipe MCP PR #5555 åpen. Nordic Financial MCP venter på Glama AAA-score.

### Alexandria MCP-server ✅
- Live: `https://alexandria.aidatanorge.no/mcp`
- TODO: `philosophy_russia.py` — ~6 000 russiske tekster: `nohup python3 philosophy_russia.py >> ~/logs/ingest_philosophy_russia.log 2>&1 &`

### XBRL historiske fikser ✅
- Re-ingest av 752 korrupte filings (2026-04-21)
- Coverage-fix: `xbrl_coverage_fix.py` — kan gjenbrukes ved fremtidige coverage-problemer
- `?`-kilde bulk-oppdatert til `source="xbrl_esef"` (10 352 vektorer)
- XBRL_FACT_TAGS utvidet (2026-05-05): `RevenueFromContractsWithCustomers` + balanse-underposter
- Scale-normalisering lagt til i `extract_xbrl_facts()` (2026-05-05)
- `xbrl_revenue_patch.py` kjørt: patcher revenue_eur for ~1144 docs via re-nedlasting av zip

### financial_summary-chunks ✅ (2026-05-05)
`financial_summary_from_xbrl.py` upserted 936 chunks (`report_type="financial_summary"`, `source="extracted_xbrl"`) med strukturerte balanse- og revenue-tall fra eksisterende XBRL-payload. Én chunk per (ticker, fiscal_year). Balanse: 99% dekning. Revenue: 66% + patch.
