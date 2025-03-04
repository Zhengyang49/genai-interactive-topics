import os
import re
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
import yaml
import openai
import logging
import json
import nltk
import time
import random
import torch
import spacy
import logging
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from process_text import *
from collections import defaultdict
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 1️⃣ Load Data --------------------

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    """Save a dictionary or list as JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# -------------------- 2️⃣ BERTopic Model Training --------------------

def load_topic_modeling(documents, config):
    """Train a BERTopic model with defined components."""
    logger.info("Initializing topic modeling...")

    # Step 1: Extract embeddings
    embedding_model = SentenceTransformer(config['embedding_model']['minilm-sm'])
    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
    plt.show()

    # Step 2: Reduce dimensionality
    umap_model = UMAP(**config['umap_model'])

    # Step 3: Cluster embeddings
    hdbscan_model = HDBSCAN(**config['hdbscan_model'])

    # Step 4: Tokenization
    vectorizer_model = CountVectorizer(**config['vectorizer_model'])

    # Step 5: Topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Initialize and run the model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        low_memory=True,
        **config['topic_model']
    )

    return topic_model, embeddings

# -------------------- 3️⃣ Topic Mapping --------------------

def map_topics_to_documents(docs, topics):
    """Map each document to its assigned topic."""
    topic_to_docs = defaultdict(list)
    for doc, topic in zip(docs, topics):
        topic_to_docs[topic].append(doc)
    return topic_to_docs

def extract_topic_sizes(topic_model, all_chunks):
    """Extracts topic assignments from documents and calculates topic sizes."""
    logger.info("Extracting document topic assignments...")

    # Get document info to extract topic assignments
    doc_info = topic_model.get_document_info(all_chunks)

    # Count the number of documents assigned to each topic
    topic_counts = doc_info['Topic'].value_counts().sort_index()

    # Convert to a dictionary for easier use
    topic_sizes = topic_counts.to_dict()

    logger.info(f"Extracted topic sizes: {topic_sizes}")
    return topic_sizes

def map_topic_sizes(topic_sizes, topic_keywords_with_labels):
    """Maps topic numbers to their corresponding labels."""
    # Remove topic -1 (outliers) and ensure all topics in topic_keywords_with_labels are included
    topic_sizes = {topic: count for topic, count in topic_sizes.items() if topic != -1}

    if len(topic_keywords_with_labels) == len(topic_sizes):
        topic_sizes_with_labels = {
            new_key: topic_sizes[old_key] for new_key, old_key in zip(topic_keywords_with_labels.keys(), topic_sizes.keys())
        }
        return topic_sizes_with_labels
    else:
        raise ValueError("Mismatch: topic_keywords_with_labels and topic_sizes must have the same length.")


# -------------------- 4️⃣ OpenAI GPT Topic Labeling --------------------

# Define a retry strategy with exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def generate_label_with_openai(message):
    """Generate a label using OpenAI's Chat API with exponential backoff."""
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use the preferred chat model
        messages=[message],
        max_tokens=50,  # Adjust token limit as needed
        temperature=0.7  # Control creativity of the response
    )
    return response.choices[0].message['content'].strip()

def generate_labels_from_keywords_openai(topic_model, documents_per_topic, topics, num_keywords=5):
    topic_labels = {}

    for topic_num in set(topics):
        if topic_num != -1:  # Skip outliers
            # Extract top keywords for the topic
            words = [word for word, _ in topic_model.get_topic(topic_num)[:num_keywords]]
            # Get a sample of documents for this topic
            sample_documents = documents_per_topic.get(topic_num, [""])[0:3]  # Use up to 3 documents for context

            # Create the message for OpenAI chat model
            message = {
                "role": "user",
                "content": f"""
                I have a topic that contains the following documents: 
                {'. '.join(sample_documents)}
                The topic is described by the following keywords: {', '.join(words)}

                Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Be as specific as possible, but don't use names of entities or countries. Use "GenAI" instead of "Generative AI" or "Genai". Capitalize the first letter of every word. Make sure it is in the following format:
                topic: <topic label>
                """
            }

            # Call the OpenAI API with exponential backoff
            try:
                label = generate_label_with_openai(message).split(":")[1].strip()
                topic_labels[topic_num] = label
            except Exception as e:
                print(f"Error generating label for topic {topic_num}: {e}")
                topic_labels[topic_num] = "Label Generation Error"

    return topic_labels

# -------------------- 5️⃣ Save Results --------------------

def save_pipeline_results(output_dir, **kwargs):
    """Save all processed results into the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for key, value in kwargs.items():
        save_json(value, os.path.join(output_dir, f"{key}.json"))

    logger.info(f"All results saved in {output_dir}")

# -------------------- 6️⃣ Run the Automated Pipeline --------------------

def run_pipeline(input_path, config_path, output_dir):
    """End-to-end automated pipeline for topic modeling."""
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()

    logger.info("Loading data and configuration...")
    all_chunks = load_json(input_path)
    config = load_config(config_path)

    logger.info(f"Total documents before processing: {len(all_chunks)}")
    processed_chunks = preprocess_chunks(all_chunks)
    processed_chunks = [doc for doc in processed_chunks if doc and isinstance(doc, str) and doc.strip() != ""]
    logger.info(f"Total documents after cleaning: {len(processed_chunks)}")
    print("Example chunk:", processed_chunks[0])
    print("Data Processing Done!", time.time()-start_time,'\n')
    
    # Train topic model
    topic_model, embeddings = load_topic_modeling(processed_chunks, config)
    topics, probs = topic_model.fit_transform(processed_chunks)

    # Extract keywords
    topic_keywords = {
        topic_num: [word for word, _ in topic_model.get_topic(topic_num)[:10]]
        for topic_num in set(topics) if topic_num != -1
    }

    # Print the top keywords for each topic
    for topic_num, keywords in topic_keywords.items():
        print(f"Topic {topic_num}: {keywords}")

    # Map topics to documents
    topic_to_docs_mapping = map_topics_to_documents(processed_chunks, topics)

    logger.info("Starting topic size extraction process...")

    # Extract topic sizes from the topic model
    topic_sizes = extract_topic_sizes(topic_model, all_chunks)

    # Generate topic labels with OpenAI
    topic_labels_gpt = generate_labels_from_keywords_openai(topic_model, topic_to_docs_mapping, topics)

    print("GPT topic labels:",topic_labels_gpt)

    # Replace topic numbers with labels
    topic_keywords_with_labels_gpt = {
        topic_labels_gpt.get(topic_num, f"Topic {topic_num}"): keywords
        for topic_num, keywords in topic_keywords.items()
    }
      
    # Assign labels to topic sizes
    topic_sizes_with_labels_gpt = map_topic_sizes(topic_sizes, topic_keywords_with_labels_gpt)
    
    # Print for verification
    print("Topic sizes with labels:", topic_sizes_with_labels_gpt)

    # Save results
    save_pipeline_results(output_dir,
        processed_document_chunks=processed_document_chunks,
        topic_keywords_with_labels_gpt=topic_keywords_with_labels_gpt,
        topic_sizes_with_labels_gpt = topic_sizes_with_labels_gpt,
        topic_to_docs_mapping=topic_to_docs_mapping,
        topic_labels_gpt=topic_labels_gpt
    )
    
    with open(os.path.join(output_dir, "topic_model.pkl"), "wb") as f:
        pickle.dump(topic_model, f)
    
    # Generate intertopic visualization
    intertopic_fig = topic_model.visualize_topics()
    
    # Check if intertopic_fig is not None before proceeding
    if intertopic_fig and hasattr(intertopic_fig, "data") and 'x' in intertopic_fig.data[0] and 'y' in intertopic_fig.data[0]:
        x_vals = np.array(intertopic_fig.data[0]['x'])
        y_vals = np.array(intertopic_fig.data[0]['y'])
    
        # Define the file path
        coords_file_path = os.path.join(output_dir, "intertopic_coords.npz")
    
        # Check if the file already exists before saving
        if not os.path.exists(coords_file_path):
            np.savez(coords_file_path, x=x_vals, y=y_vals)
            print(f"Saved intertopic coordinates to {coords_file_path}")
        else:
            print(f"File {coords_file_path} already exists. Skipping save.")
    else:
        print("intertopic_fig does not contain valid data. Skipping coordinate extraction and save.")

    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds!")


# -------------------- 7️⃣ Execute --------------------
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run BERTopic pipeline for topic modeling.")
    parser.add_argument("--type", type=str, required=True, help="Type of documents (e.g., education, finance).")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")

    # Parse arguments
    args = parser.parse_args()

    # Set OpenAI API key if needed
    openai.api_key = "xxxx"  # Consider using environment variables for security

    # Run pipeline with provided arguments
    run_pipeline(args.input, args.config, args.output)

if __name__ == "__main__":
    main()




