import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import wordnet
from extract_pdf import extract_pdf_text
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from rouge import Rouge
from evaluate import load
import bert_score

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_lg")

def is_meaningful_title(title):
    doc = nlp(title)

    #must contain at least one valid word
    valid_words = [token.text for token in doc if token.is_alpha and wordnet.synsets(token.text)]
    if not valid_words:
        return False

    #must contain a noun or a verb (common in section titles)
    if not any(token.pos_ in {"PROPN", "NOUN", "VERB"} for token in doc):
        return False

    #should not be just a function word
    if all(token.pos_ in {"DET", "ADP", "CCONJ"} for token in doc):
        return False

    #ensure numbering follows a common pattern
    if doc[0].is_digit and len(doc) > 1 and doc[1].pos_ in {"DET", "ADV"}:
        return False

    return True

def extract_sections(text):
    """Extracts section titles and their positions from text."""
    section_patterns = [
        r'^(\d+\.?\s+[A-Z][A-Za-z ]+)',   # 1. Title / 1 Title
        r'^(\d+\.\d+\.?\s+[A-Z][A-Za-z ]+)', # 1.1 Title
        r'^(I{1,3}|IV|V{1,3}|VI{1,3})\.\s+[A-Z][A-Za-z ]+',  # III. Title
        r'^[A-Z]\.\s+[A-Z][A-Za-z ]+',  # A. Title
        r'^[A-Z][A-Z ]+$'  # INTRODUCTION
    ]

    keyword_sections = {"INTRODUCTION", "METHOD", "EXPERIMENT", "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES", "ACKNOWLEDGEMENTS", "BIBLIOGRAPHY"}

    numbering_patterns = [
        r'^\d+(\.\d+)*\.?\s',  # 1, 1.1, 1.2.3
        r'^(I{1,3}|IV|V{1,3}|VI{1,3})\.\s',  # I. II. III.
        r'^[A-Z]\.\s'  # A. B. C.
    ]

    sections = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if any(re.match(pattern, line) for pattern in section_patterns) and len(line.split(' ')) < 10 or line.upper() in keyword_sections:
            if ('references' in line.lower() or 'bibliography' in line.lower() or 'acknowledgement' in line.lower()):
                break
            sections.append((i, line))

    #if a section name doesn't make sense, remove it
    sections = [sec for sec in sections if is_meaningful_title(sec[1])]

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

def detect_numbering(title):
    if match := re.match(r'^(I{1,3}|IV|V{1,3}|VI{1,3})\.\s', title):  #roman numerals
        return ("roman", match.group(0).strip())
    elif match := re.match(r'^\d+(\.\d+)*\.?\s', title):  #numeric (1, 1.1, etc.)
        value = match.group(0).strip()
        return ("numeric" + str(value.rstrip('.').count('.')), value)
    elif match := re.match(r'^[A-Z]\.\s', title):  #alphabetic (A., B., etc.)
        return ("alpha", match.group(0).strip())
    return (None, None)  #no numbering

#assume all section names have numbering
def construct_hierarchy_with_numbering(sections):
    prev_type = None

    numbering_to_prefix = {}

    for _, title in sections:
        type, value = detect_numbering(title)
        if prev_type is None:
            print(title)
            numbering_to_prefix[type] = ''
            prev_type = type
            continue
        if type not in numbering_to_prefix:
            if numbering_to_prefix[prev_type] == '':
                numbering_to_prefix[type] = '  └─── '
            else:
                numbering_to_prefix[type] = '  ' + numbering_to_prefix[prev_type]
        print(numbering_to_prefix[type] + title)
        prev_type = type

def is_valid_word(word):
    doc = nlp(word)
    lemma = doc[0].lemma_
    return word in nlp.vocab or lemma in nlp.vocab

def fix_hyphenation(text):
    #find all hyphenated words using regex
    matches = list(re.finditer(r'(\w+)-(\w+)', text))

    #process matches in reverse order (to avoid index shifting issues)
    for match in reversed(matches):
        word1, word2 = match.groups()
        joined_word = word1 + word2

        if is_valid_word(joined_word):
            #replace only if the joined word is valid
            start, end = match.span()
            text = text[:start] + joined_word + text[end:]
        else:
            print(joined_word)

    return text

#adapted from Eka's code to cover some corner cases
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(text):
    stop_words = set(stopwords.words('english'))

    #lowercase and remove URLs/special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    #text = re.sub(r'[^a-zA-Z\s]', '', text).strip()

    #Removing Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\s', '-', text)

    #Joining wordssplit by -
    text = fix_hyphenation(text)

    #Remove citations and references
    text = re.sub(r'\[\d+\]', '', text)  # Removes [12]
    text = re.sub(r'\([A-Z][^)]*,\s\d{4}[a-z]?\)', '', text)  # Removes (Smith et al., 2020)

    text = re.sub(r'[^\w\s.,!?-]', '', text)

    # Tokenize and lemmatize
    #tokens = word_tokenize(text)
    #tokens = [word.lower() if not word.isupper() else word for word in tokens]
    #tokens = [word for word in tokens if word not in stop_words]

    return text

def get_section_texts(text_lines, section_list):
    section_texts = {}

    for i in range(len(section_list) - 1):
        start_line = section_list[i][0] + 1
        section_name = section_list[i][1]
        end_line = section_list[i + 1][0] - 1

        if start_line <= end_line:
            section_texts[section_name] = " ".join(text_lines[start_line:end_line + 1])
        else:
            section_texts[section_name] = ""

    last_section_name = section_list[-1][1]
    last_section_start = section_list[-1][0]
    last_section_text = " ".join(text_lines[last_section_start:])
    last_section_text = last_section_text.split("\nReferences\n")[0]
    last_section_text = last_section_text.split("\nBibliography\n")[0]
    section_texts[last_section_name] = last_section_text

    return section_texts

def chunk_text(text, max_tokens=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        sentence_length = len(tokenized_sentence)
        length = current_length + sentence_length
        #if adding the sentence exceeds max_tokens, start a new chunk
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    #append the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_chunk(model, tokenizer, text, max_length=50):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    print('try to generate with model')
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=10,
        num_beams=8,
        length_penalty=0.5,
        early_stopping=True
    )
    res = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return res

#recursively summarize full text of each section
def summarize_full_text(text, model, tokenizer):
    chunks = chunk_text(text)
    summaries = [summarize_chunk(model, tokenizer, chunk) for chunk in chunks]
    combined_summary = " ".join(summaries)
    final_summary = summarize_chunk(model, tokenizer, combined_summary, max_length=30)

    return final_summary

def construct_markup_with_numbering(sections, section_texts, model, tokenizer):
    prev_type = ''
    ret = ''

    numbering_to_prefix = {}

    for _, title in sections:
        type, value = detect_numbering(title)
        if prev_type == '':
            numbering_to_prefix[type] = '#'
        elif type not in numbering_to_prefix:
            numbering_to_prefix[type] = '#' + numbering_to_prefix[prev_type]

        #construct summary
        summary = summarize_full_text(section_texts[title], model, tokenizer)

        ret += numbering_to_prefix[type] + ' ' + title + summary + '\n'
        prev_type = type
    return ret

#markmap generation: markmap temp.md -o markmap.html
def generate_markmap(markdown_string, output_file="markmap.html"):
    with open("temp.md", "w", encoding="utf-8") as f:
        f.write(markdown_string)

def get_summary_for_eval(text, model_name):
    #load Pegasus model & tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    sections = extract_sections(text)
    section_texts = get_section_texts(text.split('\n'), sections)
    return construct_markup_with_numbering(sections, section_texts, model, tokenizer)


df = pd.read_csv('ai_ml_papers.csv')
df_sample = df.sample(n=100, random_state=8)
#extract and append full-text
df_sample['full_text'] = df_sample['id'].apply(extract_pdf_text)
df_sample['summary_pegasus_xsum'] = df_sample['full_text'].apply(lambda x: get_summary_for_eval(x, 'google/pegasus-xsum'))
df_sample['summary_pegasus_arxiv'] = df_sample['full_text'].apply(lambda x: get_summary_for_eval(x, 'google/pegasus-arxiv'))
df_sample['summary_pegasus_pubmed'] = df_sample['full_text'].apply(lambda x: get_summary_for_eval(x, 'google/pegasus-pubmed'))
multi_scores_textrank = []
multi_scores_bart = []

COL_ROUGE1 = 'ROUGE-1'
COL_ROUGE2 = 'ROUGE-2'
COL_ROUGEL = 'ROUGE-L'
COL_BLEU = 'BLEU'
COL_BERT = 'BERTScore-F1'
COL_XSUM = ' pegasus-xsum'
COL_ARXIV = ' pegasus-arxiv'
COL_PUBMED = ' pegasus-pubmed'

df_eval[COL_ROUGE1 + COL_XSUM] = None
df_eval[COL_ROUGE1 + COL_ARXIV] = None
df_eval[COL_ROUGE1 + COL_PUBMED] = None

df_eval[COL_ROUGE2 + COL_XSUM] = None
df_eval[COL_ROUGE2 + COL_ARXIV] = None
df_eval[COL_ROUGE2 + COL_PUBMED] = None

df_eval[COL_ROUGEL + COL_XSUM] = None
df_eval[COL_ROUGEL + COL_ARXIV] = None
df_eval[COL_ROUGEL + COL_PUBMED] = None

df_eval[COL_BLEU + COL_XSUM] = None
df_eval[COL_BLEU + COL_ARXIV] = None
df_eval[COL_BLEU + COL_PUBMED] = None

df_eval[COL_BERT + COL_XSUM] = None
df_eval[COL_BERT + COL_ARXIV] = None
df_eval[COL_BERT + COL_PUBMED] = None

rouge = load("rouge")
bleu = load("bleu")

def evaluate_automatic_metrics(abstract, generated_summary):
    results = {}

    # ROUGE
    rouge_scores = rouge.compute(predictions=[generated_summary], references=[abstract])
    results["ROUGE-1"] = round(rouge_scores["rouge1"], 3)
    results["ROUGE-2"] = round(rouge_scores["rouge2"], 3)
    results["ROUGE-L"] = round(rouge_scores["rougeL"], 3)

    # BLEU
    bleu_score = bleu.compute(predictions=[generated_summary], references=[[abstract]])
    results["BLEU"] = round(bleu_score["bleu"], 3)

    # BERTScore
    P, R, F1 = bert_score.score([generated_summary], [abstract], lang="en", verbose=False)
    results["BERTScore-F1"] = round(F1[0].item(), 3)

    return results


for index, row in df_eval.iterrows():
    auto_scores_xsum = evaluate_automatic_metrics(row['abstract'], row['summary_pegasus_xsum'])
    auto_scores_arxiv = evaluate_automatic_metrics(row['abstract'], row['summary_pegasus_arxiv'])
    auto_scores_pubmed = evaluate_automatic_metrics(row['abstract'], row['summary_pegasus_pubmed'])

    df_eval.loc[index, COL_ROUGE1 + COL_XSUM] = auto_scores_xsum[COL_ROUGE1]
    df_eval.loc[index, COL_ROUGE1 + COL_ARXIV] = auto_scores_arxiv[COL_ROUGE1]
    df_eval.loc[index, COL_ROUGE1 + COL_PUBMED] = auto_scores_pubmed[COL_ROUGE1]

    df_eval.loc[index, COL_ROUGE2 + COL_XSUM] = auto_scores_xsum[COL_ROUGE2]
    df_eval.loc[index, COL_ROUGE2 + COL_ARXIV] = auto_scores_arxiv[COL_ROUGE2]
    df_eval.loc[index, COL_ROUGE2 + COL_PUBMED] = auto_scores_pubmed[COL_ROUGE2]

    df_eval.loc[index, COL_ROUGEL + COL_XSUM] = auto_scores_xsum[COL_ROUGEL]
    df_eval.loc[index, COL_ROUGEL + COL_ARXIV] = auto_scores_arxiv[COL_ROUGEL]
    df_eval.loc[index, COL_ROUGEL + COL_PUBMED] = auto_scores_pubmed[COL_ROUGEL]

    df_eval.loc[index, COL_BLEU + COL_XSUM] = auto_scores_xsum[COL_BLEU]
    df_eval.loc[index, COL_BLEU + COL_ARXIV] = auto_scores_arxiv[COL_BLEU]
    df_eval.loc[index, COL_BLEU + COL_PUBMED] = auto_scores_pubmed[COL_BLEU]

    df_eval.loc[index, COL_BERT + COL_XSUM] = auto_scores_xsum[COL_BERT]
    df_eval.loc[index, COL_BERT + COL_ARXIV] = auto_scores_arxiv[COL_BERT]
    df_eval.loc[index, COL_BERT + COL_PUBMED] = auto_scores_pubmed[COL_BERT]

df_eval_avg = df_eval[[COL_ROUGE1 + COL_XSUM, COL_ROUGE1 + COL_ARXIV, COL_ROUGE1 + COL_PUBMED, COL_BERT+COL_XSUM, COL_BERT+COL_ARXIV, COL_BERT+COL_PUBMED]]
df_eval_avg.mean()

print(f"title: {df_eval.iloc[0]['title']}")
print(f"abstract: {df_eval.iloc[0]['abstract']}")
print(f"XSUM: {df_eval.iloc[0]['summary_pegasus_xsum']}")
print(f"ARXIV: {df_eval.iloc[0]['summary_pegasus_arxiv']}")
print(f"PUBMED: {df_eval.iloc[0]['summary_pegasus_pubmed']}")