import os
import json
import time
import logging
import argparse
import numpy as np
import yaml
import openai
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from collections import defaultdict
from tenacity import retry, wait_exponential, stop_after_attempt

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_topic_modeling(documents, config):
    """Run topic modeling using BERTopic."""
    embedding_model = SentenceTransformer(config['embedding_model']['minilm-sm'])
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    umap_model = UMAP(**config['umap_model'])
    hdbscan_model = HDBSCAN(**config['hdbscan_model'])
    vectorizer_model = CountVectorizer(**config['vectorizer_model'])
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

def map_topics_to_documents(docs, topics):
    """Map topic labels to their associated documents."""
    topic_to_docs = defaultdict(list)
    for doc, topic in zip(docs, topics):
        topic_to_docs[topic].append(doc)
    return topic_to_docs

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def generate_label_with_openai(message):
    """Generate a label using OpenAI's Chat API with exponential backoff."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[message],
        max_tokens=50,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

def generate_labels_from_keywords_openai(topic_model, documents_per_topic, num_keywords=5):
    """Generate topic labels using OpenAI."""
    topic_labels = {}
    for topic_num in set(documents_per_topic.keys()):
        if topic_num != -1:  # Skip outliers
            words = [word for word, _ in topic_model.get_topic(topic_num)[:num_keywords]]
            sample_documents = documents_per_topic.get(topic_num, [""])[:3]
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
            try:
                label = generate_label_with_openai(message)
                label = label.split(":")[1].strip()
                topic_labels[topic_num] = label
            except Exception as e:
                logger.error(f"Error generating label for topic {topic_num}: {e}")
                topic_labels[topic_num] = "Label Generation Error"
    return topic_labels

def save_results(output_dir, topic_model, topics, topic_keywords, topic_sizes, topic_labels):
    """Save models and results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    topic_model.save(os.path.join(output_dir, "topic_model.pkl"))
    with open(os.path.join(output_dir, "topic_keywords.json"), "w", encoding="utf-8") as f:
        json.dump(topic_keywords, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, "topic_sizes.json"), "w", encoding="utf-8") as f:
        json.dump(topic_sizes, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, "topic_labels.json"), "w", encoding="utf-8") as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=4)
    logging.info(f"All outputs saved in {output_dir}")

def main(input_file, config_file, output_dir, openai_api_key):
    """Main function to run topic modeling."""
    start_time = time.time()
    
    documents = load_json(input_file)
    config = load_config(config_file)
    openai.api_key = openai_api_key if openai_api_key else None
    
    topic_model, embeddings = load_topic_modeling(documents, config)
    topics, _ = topic_model.fit_transform(documents)
    
    topic_keywords = {topic: [word for word, _ in topic_model.get_topic(topic)[:10]] for topic in set(topics) if topic != -1}
    topic_to_docs_mapping = map_topics_to_documents(documents, topics)
    topic_sizes = {topic: len(docs) for topic, docs in topic_to_docs_mapping.items()}
    topic_labels = generate_labels_from_keywords_openai(topic_model, topic_to_docs_mapping) if openai_api_key else topic_keywords
    
    save_results(output_dir, topic_model, topics, topic_keywords, topic_sizes, topic_labels)
    
    logging.info(f"Topic modeling completed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--openai_api_key", type=str, required=False, help="OpenAI API Key for GPT-based labeling")
    args = parser.parse_args()
    
    main(args.input_file, args.config_file, args.output_dir, args.openai_api_key)
