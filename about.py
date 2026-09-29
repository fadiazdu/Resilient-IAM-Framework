# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""Metadata for the deposited software and results (single source for headers, CITATION.cff and README)."""
TITLE = "A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management"
SUBTITLE = "Code, data, and supplemental materials"
AUTHOR = "Fredy Díaz-Durán"
ORCID = "0000-0001-5344-5466"
EMAILS = ("diazdura@ualberta.ca", "fadiazdu@uwaterloo.ca")
AFFILIATIONS = ("Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada",
                "Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada")
DOI = "10.5281/zenodo.22973335"
VERSION = "1.0.0"
LICENSES = ("Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). "
            "NBI files: public domain (DATA_NOTICE.md).")


def banner():
    return "\n".join([TITLE, SUBTITLE, f"{AUTHOR} · ORCID {ORCID} · " + " · ".join(EMAILS), *AFFILIATIONS, f"DOI: {DOI}", LICENSES])

if __name__ == "__main__":
    print(banner())
