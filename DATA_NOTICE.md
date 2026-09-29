# Data notice

The analysis uses the Rhode Island delimited files of the FHWA
National Bridge Inventory (NBI), downloaded from `https://www.fhwa.dot.gov/bridge/nbi/<year>/delimited/RI<yy>.txt` (2019-2024).
As works of the U.S. Government they are in the public domain. Every run of `run_all.py` downloads them
from FHWA into `data/raw/` and verifies each file against the SHA-256 hashes in
`data/raw/download_manifest.json` before any analysis. The Zenodo record also archives the six files
unmodified, for use with `python run_all.py --offline` only if the FHWA addresses become unavailable.

Imagery in the map figures: Esri World Imagery basemap (credit: Esri and its data providers),
used under Esri's terms of use; the basemap is not redistributed separately.
