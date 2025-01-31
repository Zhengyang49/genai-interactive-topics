import os
import re
import json
import logging
import time
import random
import pickle
import yaml
import spacy
import openai
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from tenacity import retry, wait_exponential, stop_after_attempt

# Logging setup
logger = logging.getLogger(__name__)

# Load configurations
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_nlp():
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2800000
    return nlp

# Helper functions
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def remove_ner(text, nlp, entity_labels):
    doc = nlp(text)
    lemmatized_tokens = []
    removed_entities = {}
    for token in doc:
        if token.ent_type_ in entity_labels:
            removed_entities.setdefault(token.ent_type_, []).append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)
    lemmatized_text = ' '.join(token.lemma_ for token in doc if token.ent_type_ not in entity_labels)
    return lemmatized_text.strip(), removed_entities

def preprocess_text(text, nlp, stop_words, unwanted_words, entity_labels):
    text = text.lower()
    processed_text, _ = remove_ner(text, nlp, entity_labels)
    processed_text = re.sub(r"####|[*]{2}|[-]{3,}", "", processed_text)
    processed_text = re.sub(r"\b\d+\b", "", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()
    processed_text = re.sub(r'_', ' ', processed_text)
    words = word_tokenize(processed_text)
    words = [word for word in words if word.isalpha() and word not in stop_words and word not in unwanted_words]
    return " ".join(words)

def preprocess_chunks(chunks, nlp, stop_words, unwanted_words, entity_labels):
    return [preprocess_text(chunk, nlp, stop_words, unwanted_words, entity_labels) for chunk in chunks]


# BERTopic components setup
def load_topic_modeling(documents):
    embedding_model = SentenceTransformer(config['embedding_model']['minilm-sm'])
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    umap_model = UMAP(**config['umap_model'])
    hdbscan_model = HDBSCAN(**config['hdbscan_model'])
    vectorizer_model = CountVectorizer(**config['vectorizer_model'], preprocessor=preprocess_text)
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        **config['topic_model']
    )
    return topic_model, embeddings

# Extract keywords and labels
kw_model = KeyBERT()
def generate_labels_from_keywords(topic_model, num_keywords=5):
    topic_labels = {}
    for topic_num in set(topics):
        if topic_num != -1:
            words = [word for word, _ in topic_model.get_topic(topic_num)[:10]]
            label = kw_model.extract_keywords(" ".join(words), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
            topic_labels[topic_num] = label
    return topic_labels

# Map topics to documents
def map_topics_to_documents(docs, topics):
    if len(docs) != len(topics):
        raise ValueError("Mismatch between docs and topics length")
    topic_to_docs = defaultdict(list)
    for doc, topic in zip(docs, topics):
        topic_to_docs[topic].append(doc)
    return topic_to_docs

def cluster_topics(topic_model):
    intertopic_fig = topic_model.visualize_topics()
    x_vals = np.array(intertopic_fig.data[0]['x'])
    y_vals = np.array(intertopic_fig.data[0]['y'])

    # Standardize the coordinates
    coordinates = np.vstack((x_vals, y_vals)).T
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coordinates)

    # Perform hierarchical clustering
    linkage_matrix = linkage(scaled_coords, method='ward')
    clusters = fcluster(linkage_matrix, t=1.2, criterion='distance')

    return clusters

def save_cluster_descriptions_to_json(model_dir, cluster_descriptions, output_filename="cluster_descriptions.json"):
    cluster_descriptions = {int(key): value for key, value in cluster_descriptions.items()}
    with open(os.path.join(model_dir, output_filename), "w", encoding="utf-8") as file:
        json.dump(cluster_descriptions, file, ensure_ascii=False, indent=4)
    print(f"Cluster descriptions saved to {output_filename}")

def run_topic_modeling(data_dir, openai_key=None):
    # Load configurations and initialize NLP model
    config_path = os.path.join(data_dir, "config_bertopic.yaml")
    config = load_config(config_path)
    nlp = initialize_nlp()
    stop_words = set(stopwords.words('english'))
    entity_labels = config['unwanted_entity_labels']
    unwanted_words = config['unwanted_words']

    # Load and preprocess documents
    all_chunks = load_json(os.path.join(data_dir, "documents-all.json"))
    processed_chunks = preprocess_chunks(all_chunks, nlp, stop_words, unwanted_words, entity_labels)
    processed_chunks = [doc for doc in processed_chunks if doc and isinstance(doc, str) and doc.strip() != ""]

    # Run topic modeling
    topic_model, embeddings = load_topic_modeling(processed_chunks, config)
    topics, probs = topic_model.fit_transform(processed_chunks)

    clusters = cluster_topics(topic_model)

    # Extract and generate topic labels
    kw_model = KeyBERT()
    topic_labels = generate_labels_from_keywords(topic_model)

    topic_to_docs_mapping = map_topics_to_documents(processed_chunks, topics)


    # OpenAI setup
    if openai_key:
        openai.api_key = openai_key
        @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
        def generate_label_with_openai(message):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[message],
                max_tokens=50,
                temperature=0.7
            )
            return response.choices[0].message['content'].strip()

        def generate_labels_from_keywords_openai():
            topic_labels_gpt = {}
            for topic_num in set(topics):
                if topic_num != -1:
                    words = [word for word, _ in topic_model.get_topic(topic_num)[:5]]
                    sample_documents = topic_to_docs_mapping.get(topic_num, [""])[:3]
                    message = {
                        "role": "user",
                        "content": f"I have a topic that contains the following documents: {'. '.join(sample_documents)} "
                                   f"The topic is described by the following keywords: {', '.join(words)}. "
                                   "Based on the information above, extract a short but highly descriptive topic label of at most 5 words."
                    }
                    try:
                        label = generate_label_with_openai(message).split(":")[1].strip()
                        topic_labels_gpt[topic_num] = label
                    except Exception as e:
                        topic_labels_gpt[topic_num] = "Label Generation Error"
            return topic_labels_gpt
        # Generate enriched cluster descriptions
		@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
		def generate_description_with_retry(messages):
		    response = openai.ChatCompletion.create(
		        model="gpt-4",
		        messages=messages,
		        max_tokens=400,
		        temperature=0.7
		    )
		    return response

		def generate_cluster_descriptions_GPT(clusters, topic_keywords_with_labels):
		    cluster_descriptions = {}
		    unique_clusters = np.unique(clusters)
		    for cluster_id in unique_clusters:
		        topics_in_cluster = [label for label, cluster in zip(topic_keywords_with_labels.keys(), clusters) if cluster == cluster_id]
		        keywords = [", ".join(topic_keywords_with_labels[label]) for label in topics_in_cluster]
		        messages = [
		            {
		                "role": "user",
		                "content": (
		                    f"Here are the topics in Cluster {cluster_id}: {', '.join(topics_in_cluster)}.\n"
		                    f"Each topic is characterized by the following keywords:\n" +
		                    "".join(f"- {label}: {', '.join(topic_keywords_with_labels[label])}\n" for label in topics_in_cluster) +
		                    "\nPlease generate a cohesive and detailed description of this cluster, highlighting the main themes, distinctions, and how these topics are interrelated."
		                )
		            }
		        ]
		        try:
		            response = generate_description_with_retry(messages)
		            description = response.choices[0].message['content'].strip()
		        except Exception as e:
		            description = f"Error generating description: {str(e)}"
		        cluster_descriptions[cluster_id] = f"Cluster {cluster_id} includes topics: {', '.join(topics_in_cluster)}. " + description
		    return cluster_descriptions

        topic_labels_gpt = generate_labels_from_keywords_openai()
        cluster_descriptions = generate_cluster_descriptions_GPT(clusters, topic_labels_gpt)
    else:
        topic_labels_gpt = {}
        cluster_descriptions = {}

    # Finalize and save
	final_labels = {topic_num: labels[0][0] for topic_num, labels in topic_labels.items()}
	def replace_key_with_labels(final_labels):
	    return {final_labels[i]: keywords for i, (label, keywords) in enumerate(topic_keywords.items())}

	topic_keywords_with_labels_gpt = replace_key_with_labels(topic_labels_gpt)
	doc_info = topic_model.get_document_info(all_chunks)
	topic_counts = doc_info['Topic'].value_counts().sort_index()
	topic_sizes = {topic: count for topic, count in topic_counts.items() if topic != -1}

	def topic_size(topic_sizes, topic_keywords_with_labels):
	    if len(topic_keywords_with_labels) == len(topic_sizes):
	        return {new_key: topic_sizes[old_key] for new_key, old_key in zip(topic_keywords_with_labels.keys(), topic_sizes.keys())}
	    else:
	        raise ValueError("Mismatch between topic_keywords_with_labels and topic_sizes length")


	topic_sizes_with_labels_gpt = topic_size(topic_sizes, topic_labels_gpt)

    # Save results
    data_dir = "saved_topic_model"
    os.makedirs(data_dir, exist_ok=True)
    save_cluster_descriptions_to_json(model_dir, cluster_descriptions)
    with open(os.path.join(data_dir, "topic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)
    with open(os.path.join(data_dir, "topic_keywords_with_labels.json"), "w") as f:
        json.dump(topic_labels, f)
    with open(os.path.join(data_dir, "topic_keywords_with_labels_gpt.json"), "w") as f:
        json.dump(topic_labels_gpt, f)
    with open(os.path.join(data_dir, "topic_sizes_with_labels_gpt.json"), "w") as f:
    json.dump(topic_sizes_with_labels_gpt, f)
    print("Topic modeling completed and results saved.")

if __name__ == "__main__":
    run_topic_modeling(data_dir, openai_key=None)














