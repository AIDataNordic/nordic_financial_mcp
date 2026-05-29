# DigiDollar Migration Notes
**Sist oppdatert:** 2026-05-29
**Status:** ✅ RC43 (testnet25) kjører — oracle 11 aktiv og i konsensus

---

## System Info
- **Server:** Minisforum (mini-hal@inisforum)
- **Laptop:** hallvardo@hallvardo-Latitude-E7440 (SSH-tilgang til server)
- **Starlink:** CGNAT — innkommende P2P-tilkoblinger fungerer ikke, kun utgående

---

## Hva er gjort ✅

1. **Stoppet gammel node** — systemd-tjeneste: `digibyte-testnet.service`
2. **Installert RC30-binærer** til `/usr/local/bin/`
   - Versjon bekreftet: `DigiByte version v9.26.0rc30-g5188782b056259e2d8018cd69b276764e5c4c55b`
3. **Oppdatert `~/.digibyte/digibyte.conf`:**
   ```ini
   server=1
   listen=1
   txindex=1
   digidollar=1
   digidollarstatsindex=1
   debug=digidollar
   debug=net
   rpcuser=digibyte
   rpcpassword=digibyte123
   [test]
   testnet=1
   addnode=oracle1.digibyte.io:12030
   walletcrosschain=1
   rpcallowip=127.0.0.1
   rpcbind=127.0.0.1
   rpcport=14026
   port=12030
   ```
4. **Firewall:** Ikke åpnet port (Starlink CGNAT gjør det unødvendig)
5. **Startet RC30-node** — opprettet `~/.digibyte/testnet23/`
6. **Kopiert oracle wallet** fra testnet21 til testnet23:
   ```
   ~/.digibyte/testnet21/wallets/oracle/wallet.dat
   → ~/.digibyte/testnet23/wallets/oracle/wallet.dat
   ```
7. **Patchet application_id** i wallet.dat:
   - Gammel: `0xFDD2B9E3` (RC28/RC29, testnet21)
   - Ny: `0xFDD2B9E4` (RC30, testnet23)
   - Offset: byte 68, big-endian uint32
   - Backup: `wallet.dat.pre-patch.bak`
8. **Node synkroniserer** — blokkhøyde ~1843 ved siste sjekk (2026-04-20 16:00)
9. **Wallet laster** — `loadwallet "oracle"` returnerer `{"name": "oracle"}`
   - NB: Wallet lastes **ikke** automatisk ved oppstart — må lastes manuelt etter hver restart

---

## Problemet — LØST (root cause funnet) ✅

~~RC30 klarer ikke starte oracle med eksisterende RC28-nøkkel.~~

**Rotårsak: RC30 chainparams har fortsatt den gamle pubkeyen for oracle 11.**

Kronologi:
- **27. mars:** Opprinnelig oracle 11-nøkkel generert på laptop (`hallvardo-Latitude-E7440`) i `~/digibyte-testnet/wallets/oracle/` — pubkey `03dfcb956f9e6f8ceea00b067176baa118ba8f0fbdb171a821a362af19234e64bd`
- **17. april:** Konstatert at privkey til `03dfcb956...` ikke var tilgjengelig (generert på Raspiblitz, aldri overført til Ubuntu-server). Ny nøkkel generert på server: `024ef063a67b35295e9eaaa9251bc7f0effbceaedc8e9bc92504b0da832744ca08`. Bedt Jared om å oppdatere chainparams.
- **RC30 sluppet** uten at chainparams ble oppdatert — oracle 11 har fortsatt `03dfcb956...`

RC30 chainparams (bekreftet):
```cpp
{11, ParsePubKey("03dfcb956f9e6f8ceea00b067176baa118ba8f0fbdb171a821a362af19234e64bd"), "oracle12.digibyte.io:12030", true}, // hallvardo
```

`startoracle 11` feiler fordi lokal privkey produserer `024ef063...`, som ikke matcher chainparams-pubkeyen `03dfcb956...`. Dette er et pubkey-mismatch, ikke et format- eller migrasjonsproblem.

---

## Diagnose (utført 2026-04-20)

### Verifisert: node kjører testnet23
- `digibyted` kjører på port 14026
- `testnet23/debug.log` har fersk aktivitet
- Chain: `test`, blokkhøyde 1843

### Verifisert: wallet.dat-format er uendret mellom RC28 og RC30
Struktursammenlikning mellom RC28-generert og RC30-generert `oraclekey`:
- Begge er 215 bytes
- Identisk ASN.1 DER-struktur (SEC1/RFC5915)
- Ingen migrasjonsendring i nøkkelformat

### Verifisert: alle wallet.dat-filer sjekket
| Fil | Oracle pubkey |
|-----|--------------|
| `testnet23/wallets/oracle` | `024ef063...` (aktiv nøkkel) |
| `testnet23/wallets/oracle_t21` | `024ef063...` (kopi) |
| `testnet21/wallets/oracle` | `024ef063...` (kopi) |
| `testnet23/wallets/TempWallet` | `033b6494...` (ikke oracle 11) |
| `testnet23/wallets/oracle_rc30_test` | `03a6dc62...` (testgenerert ID 28) |
| `testnet19/wallets/oracle` | `03ef4ead...` (gammel testnet19) |
| `testnet21/wallets/test123` | ingen oracle-nøkkel |
| `testnet19/wallets/testwallet` | ingen oracle-nøkkel |
| Laptop: `Downloads/wallet.dat` | ingen oracle-nøkkel |
| Laptop: `testnet19/wallets/oracle` | ingen oracle-nøkkel |

Privkey til `03dfcb956...` finnes ikke på noen kjent maskin.

### Avkreftet
- ~~RC29-backup hadde migrert nøkkelformat~~ — backup er RC28, ingen endring i value-lengde
- ~~Wallet ikke lastet~~ — wallet lastes korrekt, men ikke automatisk
- ~~Port-konflikt~~ — RC28-prosess blokkerte port 14026, løst med `sudo kill <pid>`
- ~~Feil tabell i SQLite~~ — kun én tabell (`main`), korrekt lest
- ~~Nøkkelformat-problem~~ — RC28 og RC30 bruker identisk format (215 bytes, ASN.1 DER)
- ~~Migrasjonsproblem~~ — rotårsak er pubkey-mismatch i chainparams, ikke migrasjonsformat

---

## CPU Mining — 2026-05-28/29

### Mining-adresse
```
dgbt1q4u3n4h9gyahq2qljvcjtupv37wfqk5lgd0xtsk
```
Opprettet via `getnewaddress` i oracle-wallet (2026-05-28).

### Status per 2026-05-29
| Metrikk | Verdi |
|---------|-------|
| Blokker minet | 15 |
| Total belønning | ~16 177 DGB (immature) |
| Status | Immature — trenger 100+ bekreftelser |
| Wallet-balanse (total) | 361 870 DGB |

### Mining-loop (CPU, testnet)
```bash
while true; do
  digibyte-cli -testnet generatetoaddress 1 dgbt1q4u3n4h9gyahq2qljvcjtupv37wfqk5lgd0xtsk && echo "Block mined $(date)"
done
```

### Sjekk mining-status
```bash
# Transaksjoner til mining-adresse
digibyte-cli -testnet -rpcwallet=oracle listtransactions "*" 100 0 true | python3 -c "
import sys, json
txs = json.load(sys.stdin)
mine = [t for t in txs if t.get('address') == 'dgbt1q4u3n4h9gyahq2qljvcjtupv37wfqk5lgd0xtsk']
print(f'Blokker: {len(mine)}, Total: {sum(t[\"amount\"] for t in mine):.2f} DGB')
"
```

---

## RC43 Migration — 2026-05-28 ✅

### Hva ble gjort
1. Lastet ned prebuilt `digibyte-9.26.0-rc43-x86_64-linux-gnu.tar.gz` fra GitHub releases
2. Backup av oracle-wallet: `~/oracle_backup_rc41_20260528.dat`
3. Stoppet node: `sudo systemctl stop digibyte-testnet`
4. Kopiert binærer til `~/digibyte/bin/`
5. Startet node: `sudo systemctl start digibyte-testnet`
6. Lastet wallet og startet oracle — bekreftet `success: true`, `status: running`

**Ingen wallet-patching nødvendig** — RC43 bruker samme testnet25-kjede som RC41/RC42 (ingen reset).

**Viktigste endringer i RC43:** Stabiliseringsrelease etter RC42. Hardnet rapid DigiDollar mint/send/redeem wallet-state. DigiDollar Qt-UI-forbedringer (labels, dialogs, tab). `createoraclekey` kan nå kjøres før DigiDollar-aktivering for å generere mainnet oracle-nøkler.

---

## RC36 Migration — 2026-05-13 ✅

### Hva ble gjort
1. Lastet ned prebuilt `digibyte-9.26.0-rc36-x86_64-linux-gnu.tar.gz` fra GitHub releases
2. Stoppet node: `sudo systemctl stop digibyte-testnet`
3. Kopiert binærer til `~/digibyte/bin/`
4. Startet node: `sudo systemctl start digibyte-testnet`
5. Lastet wallet og bekreftet `in_consensus: true`, `last_price_micro_usd: 3925`

**Ingen wallet-patching nødvendig** — RC36 bruker samme testnet24-kjede som RC35 (ingen reset).

**Viktigste endringer i RC36:** MuSig2-partial signatures bindes nå til eksplisitt sesjonskontekst (pris/timestamp + signer bitmap + nonce set + epoch + chain). Partial sigs bufres per epoch+kontekst med per-kontekst/per-epoch caps (DoS-guard).

---

## RC35 Migration — 2026-05-12 ✅

### Hva ble gjort
1. Lastet ned `digibyte-9.26.0-rc35.tar.gz` (kildekode)
2. Kompilert fra kilde: `./autogen.sh && ./configure --without-gui --disable-tests --disable-bench && make -j4` i `~/digibyte-9.26.0-rc35/`
3. Stoppet node: `sudo systemctl stop digibyte-testnet`
4. Kopiert binærer til `~/digibyte/bin/`
5. Startet node: `sudo systemctl start digibyte-testnet`
6. Lastet wallet og bekreftet `in_consensus: true`, `last_price_micro_usd: 3909`

**Ingen wallet-patching nødvendig** — RC35 bruker samme testnet24-kjede som RC34 (ingen reset).

## RC34 Migration — 2026-05-12 ✅

### Hva ble gjort
1. Lastet ned `digibyte-9.26.0-rc34.tar.gz` (kildekode)
2. Kompilert fra kilde: `./configure && make -j$(nproc)` i `~/digibyte-9.26.0-rc34/`
3. Kopiert binærer til `~/digibyte/bin/`
4. Backup av wallet: `~/.digibyte/testnet23/wallets/wallets.bak_rc28`
5. Kopiert oracle-wallet til testnet24: `~/.digibyte/testnet23/wallets/oracle/ → ~/.digibyte/testnet24/wallets/oracle/`
6. Patchet `wallet.dat` application_id til testnet24 magic bytes (`0xFEC4B7E5`, offset 68, big-endian)
7. Oppdatert `digibyte-testnet.service` til å peke på `~/digibyte/bin/`
8. Aktivert systemd-tjeneste: `sudo systemctl enable --now digibyte-testnet`
9. Lastet wallet og startet oracle — bekreftet `in_consensus: true`

### Testnet24 nettverksdetaljer
| Parameter | Verdi |
|-----------|-------|
| Datamappe | `~/.digibyte/testnet24/` |
| P2P-port | `12031` |
| RPC-port | `14026` |
| Magic bytes | `0xFEC4B7E5` |
| Aktiveringsblokk DigiDollar | 600 |

### Pubkey-status
RC34 chainparams har korrekt pubkey for oracle 11: `024ef063a67b35295e9eaaa9251bc7f0effbceaedc8e9bc92504b0da832744ca08` ✅

### Wallet-patching ved fremtidig RC-oppgradering
Når ny RC introduserer nytt testnet, må wallet.dat patches med nye magic bytes:
```bash
# Finn magic bytes i src/kernel/chainparams.cpp (pchMessageStart for nytt testnet)
# Patch byte 68 i wallet.dat:
python3 -c "import struct; f=open('wallet.dat','r+b'); f.seek(68); f.write(struct.pack('>I', 0xNEWMAGIC)); f.close()"
```

---

## Nyttige kommandoer

```bash
# Sjekk node-status
sudo systemctl status digibyte-testnet

# Start/stopp/restart node
sudo systemctl start digibyte-testnet
sudo systemctl stop digibyte-testnet
sudo systemctl restart digibyte-testnet

# Sjekk om gammel prosess blokkerer porten
sudo ss -tlnp | grep 14026

# Last wallet (må gjøres manuelt etter hver restart)
digibyte-cli -testnet loadwallet "oracle"

# Start oracle (fungerer når chainparams er oppdatert med ny pubkey)
digibyte-cli -testnet -rpcwallet=oracle startoracle 11

# Sjekk blokkhøyde
digibyte-cli -testnet getblockchaininfo | grep blocks

# Sjekk oracle-status i nettverket
digibyte-cli -testnet getoracles

# Sjekk hvilken pubkey som er registrert for oracle 11
digibyte-cli -testnet getoracles | grep -A10 '"oracle_id": 11'

# Sjekk tilgjengelige oracle-kommandoer
digibyte-cli -testnet help | grep -i oracle

# Live logg
tail -f ~/.digibyte/testnet23/debug.log

# Sjekk alle wallets
digibyte-cli -testnet listwalletdir

# Ekstraher pubkey fra wallet-kopi (kjør offline, ikke på aktiv wallet)
python3 -c "import sqlite3; conn = sqlite3.connect('/home/mini-hal/.digibyte/testnet23/wallets/oracle_t21/wallet.dat'); cursor = conn.cursor(); cursor.execute('SELECT hex(value) FROM main WHERE hex(key) LIKE \"%6F7261636C656B6579%\"'); row = cursor.fetchone(); pubkey = bytes.fromhex(row[0])[-33:].hex() if row else 'ikke funnet'; print('Pubkey:', pubkey); conn.close()"

# Backup av aktiv wallet (omgår database locked)
digibyte-cli -testnet -rpcwallet=oracle backupwallet /tmp/oracle_backup.dat
```

---

## Relevante filer på server
```
~/.digibyte/digibyte.conf                          # Konfig
~/.digibyte/testnet21/wallets/oracle/wallet.dat    # Original RC28-wallet (024ef063...)
~/.digibyte/testnet23/wallets/oracle/wallet.dat    # Patchet RC30-wallet (aktiv, 024ef063...)
~/.digibyte/testnet23/wallets/oracle_t21/          # Kopi av testnet21-wallet for analyse
~/.digibyte/testnet23/debug.log                    # RC30 logg
~/.digibyte/testnet21/debug.log                    # RC28 logg (historikk)
/usr/local/bin/digibyted.rc29.bak                  # Backup — NB: er egentlig RC28, ikke RC29!
```

## Nøkkeloversikt
| Nøkkel | Status |
|--------|--------|
| `03dfcb956...` | Opprinnelig oracle 11-nøkkel. Hardkodet i RC30 chainparams. Privkey tapt (lå på Raspiblitz). |
| `024ef063...` | Ny oracle 11-nøkkel. Generert 2026-04-17 på Ubuntu-server. Klar i wallet. Venter på chainparams-oppdatering. |

---

## Gitter-kanal
```
https://app.gitter.im/#/room/#digidollar:gitter.im
```
