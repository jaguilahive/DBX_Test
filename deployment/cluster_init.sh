#!/bin/bash
set -e

apt-get update -qq
apt-get install -y -qq tesseract-ocr tesseract-ocr-eng antiword

pip install -q \
    PyMuPDF==1.24.3 \
    pdfplumber==0.11.0 \
    pytesseract==0.3.10 \
    python-docx==1.1.0 \
    python-magic==0.4.27 \
    xlrd==2.0.1 \
    tiktoken==0.7.0 \
    chardet==5.2.0 \
    pyyaml>=6.0 \
    openpyxl>=3.1.2
