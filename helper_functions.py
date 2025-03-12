import requests
import fitz
import pandas as pd
from io import BytesIO

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

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
    
#tokenization and text cleaning
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase and remove URLs/special characters
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    #text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    
    #Removing Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    #Joining wordssplit by -
    text = re.sub(r'(\w+)-\s(\w+)', '\1\2', text)
    
    #Remove citations and references
    text = re.sub(r'\[\d+\]', '', text)  # Removes [12]
    text = re.sub(r'\(.*?et al\.,\s*\d{4}\)', '', text)  # Removes (Smith et al., 2020)
    ref = "references"
    text = text.split(ref)[0].strip() if ref in text else text
    
    #Remove content before introduction
    intro = "introduction"
    text = text.split(intro)[1].strip() if intro in text else text
    
    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Join sentences while keeping within the 1024-token limit
def chunk_text(sentences, tokenizer, max_tokens=1024):
    chunks = []
    current_chunk = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        if len(tokenized_sentence) + sum(len(tokenizer.tokenize(s)) for s in current_chunk) < max_tokens:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks