"""Checks that data/loan_approval_dataset.csv is byte-identical to the file used in the
independent assessment (SHA-256 printed in its Section 2), and reports its class counts.

An earlier copy of this repository held the same table with a trailing comma appended to
every line (SHA-256 996bfe05...e0ca); removing those commas gives exactly the assessor's bytes.
"""
from pathlib import Path
import hashlib
import pandas as pd

ASSESSOR_SHA256 = "4b5cd093d178378f4cfa8c107adb6e599b88be9d8a3b51f3b99c0d5914154e54"
PREVIOUS_TRAILING_COMMA_SHA256 = "996bfe05f4bad6bbd104ac7bb8e0329dcc0ece7c16c6956294154795d506e0ca"
path = Path(__file__).resolve().parents[1] / "data" / "loan_approval_dataset.csv"
raw = path.read_bytes()
h = hashlib.sha256(raw).hexdigest()
print("Dataset:", path)
print("SHA-256:", h)
print("Assessor's SHA-256:", ASSESSOR_SHA256)
if h == PREVIOUS_TRAILING_COMMA_SHA256:
    fixed = b"\r\n".join(l[:-1] if l.endswith(b",") else l for l in raw.split(b"\r\n"))
    print("This is the earlier trailing-comma copy; without the commas its SHA-256 is",
          hashlib.sha256(fixed).hexdigest())
if h != ASSESSOR_SHA256:
    raise SystemExit("Dataset is not byte-identical to the assessor's file")
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]
counts = df["loan_status"].str.strip().value_counts()
print(f"Rows {len(df)}, columns {df.shape[1]}, missing {int(df.isna().sum().sum())}")
print(f"Approved {counts['Approved']} ({counts['Approved']/len(df):.3%}), Rejected {counts['Rejected']} ({counts['Rejected']/len(df):.3%})")
print("Checksum verified: byte-identical to the assessor's file.")
