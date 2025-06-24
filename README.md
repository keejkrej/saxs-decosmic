# XRD Decosmic

A tool for removing high energy background from XRD 2D images.

## Installation

```bash
git clone https://github.com/keejkrej/xrd-decosmic.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ag-nickel/xrd-decosmic.git
cd xrd-decosmic
unset PIP_REQUIRE_VIRTUALENV
pip install -r requirement.txt --no-user
pip install . --no-user --no-deps
```

## Usage

- activate your python environment in which the program is installed
- copy [process.py](scripts/process.py) to your folder
- change **INPUT_FILE**, **OUTPUT_DIR**, **OUTPUT_PREFIX**, **TH_DONUT**, **TH_MASK**, **TH_STREAK**, **WIN_STREAK**, **EXP_DONUT**, **EXP_STREAK** accordingly and run