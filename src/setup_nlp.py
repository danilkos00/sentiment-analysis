import nltk
import spacy
from spacy.cli import download as spacy_download

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy_download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download('punkt')
