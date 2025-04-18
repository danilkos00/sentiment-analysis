import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.cli import download as spacy_download

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy_download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

nltk_packages = ['punkt', 'stopwords']

for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package)
