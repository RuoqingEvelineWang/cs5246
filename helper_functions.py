import requests
import fitz
import pandas as pd
from io import BytesIO

from nltk.corpus import wordnet

import spacy

import re


def pad_arxiv_id(arxiv_id):
    parts = str(arxiv_id).split(".")
    prefix = parts[0].zfill(4)
    return f"{prefix}.{parts[1]}"

def extract_pdf_text(arxiv_id):
    padded_arxiv_id = pad_arxiv_id(arxiv_id)

    url = f"https://arxiv.org/pdf/{padded_arxiv_id}.pdf"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)

        pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
        full_text = "\n".join([page.get_text("text") for page in pdf_doc])

        return full_text
    
    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {e}"
    except Exception as e:
        return f"Error extracting text: {e}"

nlp = spacy.load("en_core_web_sm")

def extract_sections(text):

    keyword_sections = {"ABSTRACT","INTRODUCTION", "METHOD", "EXPERIMENT", "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES", "ACKNOWLEDGEMENTS", "BIBLIOGRAPHY"}

    numbering_patterns = [
        r'^\d+(\.\d+)*\.?\s',  # 1, 1.1, 1.2.3
        r'^(I{1,3}|IV|V{1,3}|VI{1,3})\.\s',  # I. II. III.
        r'^[A-Z]\.\s'  # A. B. C.
    ]

    sections = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if line.upper() in keyword_sections:
            sections.append((i, line))

    #if there is an existing numbering pattern, remove other titles as noise
    numbering_pattern_count = 0
    for section in sections:
        for pattern in numbering_patterns:
            if re.match(pattern, section[1]):
                numbering_pattern_count += 1
    if numbering_pattern_count > 5:
        for section in sections[:]:
            matched = False

            for pattern in numbering_patterns:
                if re.match(pattern, section[1]):
                    matched = True
            if not matched:
                sections.remove(section)

    return sections