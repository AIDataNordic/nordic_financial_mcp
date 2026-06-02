"""One-shot x402 payment test against Alfred /report endpoint."""
import os
import sys

from eth_account import Account
from x402.client import x402ClientSync
from x402.mechanisms.evm.exact import register_exact_evm_client
from x402.mechanisms.evm.signers import EthAccountSigner
import requests
from x402.http.clients.requests import x402_http_adapter

PRIVATE_KEY = os.environ["WALLET_PRIVATE_KEY"]
URL = "https://alfred.aidatanorge.no/report?company=Equinor"

account = Account.from_key(PRIVATE_KEY)
print(f"Wallet: {account.address}")

signer = EthAccountSigner(account)
client = x402ClientSync()
register_exact_evm_client(client, signer)

session = requests.Session()
session.mount("https://", x402_http_adapter(client))

print(f"Making paid request to {URL} ...")
response = session.get(URL)
print(f"Status: {response.status_code}")
print(response.text[:2000])
