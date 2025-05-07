# XRD Decosmic
A tool for removing high energy background from XRD 2D images.
## Installation
```bash
git clone https://github.com/keejkrej/xrd-decosmic.git
cd xrd-decosmic
pip install -e .
```
## Usage
```bash
python -m xrd_decosmic.cli \
--input /path/to/first_filename.tif \
--output /path/to/output/ \
--th-donut 15 \
--th-mask 0.05 \
--th-streak 3 \
--win-streak 3 \
--exp-donut 9 \
--exp-streak 3
```
