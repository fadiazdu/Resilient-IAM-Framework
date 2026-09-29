# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
download_data.py - fetch the public FHWA National Bridge Inventory files
=======================================================================
Downloads the Rhode Island (state code 44) delimited NBI files for 2019-2024
directly from FHWA's public website and saves them unchanged as
data/raw/RI{year}.csv. No other data enter the model: every downstream input
(inventory, inspection intervals, deterioration matrix) is computed from these
files by data_prepare.py and deterioration.py.

Sources, in order of preference for each year:
  1. Per-state delimited file:
     https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/RI{yy}.txt
  2. Fallback, the national delimited archive filtered to state 44:
     https://www.fhwa.dot.gov/bridge/nbi/{year}del.zip
     (the route used by the original build_matrix.py)

Verification: each file's SHA-256 is compared with the reference hash in
data/raw/download_manifest.json (the files used for the published results).
A mismatch is reported, not fatal: FHWA occasionally revises archived files,
and a revised file may change the results. Every run writes
data/raw/download_log.json with URL, size, hash, and match status.

Usage:
  python download_data.py            # download all years (skips verified files)
  python download_data.py --force    # re-download even if a verified copy exists
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, time, zipfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
YEARS = range(2019, 2025)
STATE_CODE, STATE_ABBR = "44", "RI"
# Base URL is overridable only so the download logic can be tested offline.
BASE = os.environ.get("IAM_NBI_BASE", "https://www.fhwa.dot.gov/bridge/nbi")
HEADERS = {"User-Agent": "Mozilla/5.0 (research reproduction; ResilientIAM)"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _reference() -> dict[int, dict]:
    p = RAW / "download_manifest.json"
    return {e["year"]: e for e in json.load(open(p))} if p.exists() else {}


def _state_file(year: int) -> tuple[bytes, str]:
    url = f"{BASE}/{year}/delimited/{STATE_ABBR}{str(year)[2:]}.txt"
    return _get(url), url


def _national_filtered(year: int) -> tuple[bytes, str]:
    """Fallback: national delimited archive, keeping the header and state-44 rows."""
    url = f"{BASE}/{year}del.zip"
    z = zipfile.ZipFile(io.BytesIO(_get(url, timeout=600)))
    name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)   # the data file
    lines = z.read(name).splitlines(keepends=True)
    head, body = lines[0], lines[1:]
    keep = [l for l in body if l.split(b",")[0].strip(b"' \"") == STATE_CODE.encode()]
    if not keep:
        raise RuntimeError(f"no state {STATE_CODE} rows in {url}")
    return head + b"".join(keep), url + f" (filtered to state {STATE_CODE})"


def download(force: bool = False) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    ref, log = _reference(), []
    for year in YEARS:
        out = RAW / f"RI{year}.csv"
        r = ref.get(year, {})
        data, url, errors = None, None, []
        for fetch in (_state_file, _national_filtered):
            try:
                data, url = fetch(year); break
            except Exception as e:                      # try the next public source
                errors.append(f"{fetch.__name__}: {type(e).__name__}: {e}")
        if data is None:
            raise RuntimeError(f"{year}: all FHWA sources failed:\n  " + "\n  ".join(errors))
        out.write_bytes(data)
        h = _sha(data); match = (h == r.get("sha256")) if r else None
        flag = "matches reference" if match else ("DIFFERS from reference (file revised or fallback source)" if r else "no reference hash")
        print(f"  {year}: {len(data):,} bytes from {url} -> {flag}")
        log.append(dict(year=year, url=url, bytes=len(data), sha256=h, matches_reference=match, source="downloaded"))
    json.dump(dict(downloaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=log),
              open(RAW / "download_log.json", "w"), indent=1)
    n_ok = sum(1 for e in log if e["matches_reference"])
    print(f"  {n_ok} of {len(log)} files identical to the reference files used for the reported results")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even when a verified copy exists")
    download(ap.parse_args().force)
