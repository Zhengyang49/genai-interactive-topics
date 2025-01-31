import os
import json
import re
import spacy
import random
import numpy as np
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner"])
stop_words = set(stopwords.words('english'))
nlp.max_length = 2800000
random.seed(42)
np.random.seed(42)

custom_stopwords = {'references', 'appendix', 'glossary', 'table of contents', 'acknowledgments',
                    'disclosure statement', 'author', 'contact', 'executive summary', 'introduction',
                    'funding', 'citation', 'endnotes', 'notes', 'abstract', 'bibliography'}

def remove_ner(text):
    """Remove named entities from text and perform lemmatization."""
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        if not token.ent_type_:
            lemmatized_tokens.append(token.lemma_)
    return " ".join(lemmatized_tokens)

def preprocess_text(text):
    """Perform text preprocessing including NER removal, lemmatization, and stopword filtering."""
    text = text.lower()
    processed_text = remove_ner(text)
    processed_text = re.sub(r"####|[*]{2}|[-]{3,}", "", processed_text)
    processed_text = re.sub(r"\b\d+\b", "", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()
    processed_text = re.sub(r'_', ' ', processed_text)
    words = word_tokenize(processed_text)
    words = [word for word in words if word.isalpha() and word not in stop_words and word not in custom_stopwords]
    return " ".join(words)

def preprocess_chunks(input_file, output_file):
    """Load JSON text data, preprocess each chunk, and save results."""
    with open(input_file, "r", encoding="utf-8") as file:
        text_chunks = json.load(file)
    
    processed_chunks = [preprocess_text(chunk) for chunk in text_chunks]
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_chunks, file, ensure_ascii=False, indent=4)
    
    print(f"Preprocessed text saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    
    preprocess_chunks(args.input_file, args.output_file)
