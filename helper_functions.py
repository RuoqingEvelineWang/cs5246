import requests
import fitz
import pandas as pd
from io import BytesIO

from collections import Counter
from nltk.corpus import stopwords
import string

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
    
    
    #