"""
verify_response.py
Regenerates every number quoted in the response from the deposited outputs (make_numbers.py)
and checks that each value the builder used appears, exactly as formatted, in the document.

    python response/verify_response.py [response.docx]
"""
import json, re, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
docx = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "Response_to_Reproducibility_Assessment_FINAL.docx")
tmp = HERE / "N_recomputed.json"
subprocess.run([sys.executable, str(HERE / "make_numbers.py"), str(tmp)], check=True)
N = json.load(open(tmp, encoding="utf-8"))
stored = json.load(open(HERE / "N.json", encoding="utf-8"))
used = json.load(open(HERE / "used_keys.json"))
drift = [k for k in used if N.get(k) != stored.get(k)]
xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
paras = ["".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p_, flags=re.S))
         for p_ in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
text = "\n".join(paras).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
missing = [(k, N[k]) for k in used if k not in N or N[k] not in text]
bad = [w for w in ("undefined", "NaN", "null", "[object") if w in text]
print(f"Checked {len(used)} values against {Path(docx).name}")
for k in drift: print("  CHANGED since build:", k, stored.get(k), "->", N.get(k))
for k, v in missing: print("  NOT FOUND", k, v)
if bad: print("  invalid tokens:", bad)
tmp.unlink()
if missing or bad or drift: sys.exit(1)
print("PASS: every recomputed value appears in the response exactly as reported.")
