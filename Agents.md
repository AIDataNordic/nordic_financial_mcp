## Runtime-begrensninger

Denne kodebasen bruker live tjenester på localhost (Qdrant :6333, Alfred :8006,
MCP :8003). Du kan ikke nå disse fra sandbox. Konsekvenser:
- Ikke bekreft Qdrant-resultater uten å si eksplisitt at du ikke har kjørt spørringen
- Ikke påstå at ticker-data finnes/mangler uten at brukeren har verifisert det
- Be brukeren kjøre kommandoer og lime inn output når du trenger faktisk systemdata
