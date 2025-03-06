import os
import json
import pickle
import numpy as np
import argparse
from collections import defaultdict

def load_models(base_dir, categories):
    """
    Load BERTopic models and topic keyword mappings.
    """
    topic_models = {}
    topic_keywords_mappings = {}

    for category, sub_dir in categories.items():
        model_dir = os.path.join(base_dir, 'model_info', sub_dir)
        topic_model_path = os.path.join(model_dir, "topic_model.pkl")
        topic_keywords_path = os.path.join(model_dir, "topic_keywords_with_labels_gpt.json")

        # Load BERTopic model
        with open(topic_model_path, "rb") as f:
            topic_models[category] = pickle.load(f)

        # Load topic keywords with labels
        with open(topic_keywords_path, "r") as f:
            topic_keywords_mappings[category] = json.load(f)

    return topic_models, topic_keywords_mappings


def find_topic_label(topic_id, category):
    """
    Get topic label from topic_keywords_with_labels_gpt.json based on topic ID.
    """
    topic_info = topic_models[category].get_topic(topic_id)
    topic_keywords = [word for word, _ in topic_info] if topic_info else []

    if not topic_keywords:
        return "Unknown Topic", []

    # Find the best matching topic label
    for label, keywords in topic_keywords_mappings[category].items():
        if any(keyword in topic_keywords for keyword in keywords):
            return label, keywords

    return "Miscellaneous GenAI Topics", topic_keywords


def process_all_chunks(chunks, topic_models, topic_keywords_mappings):
    """
    Process all chunks using the "all" model and get keywords and topics.
    Saves intermediate results to a JSON file.
    """
    processed_chunks_all = []

    all_cleaned_texts = [chunk["cleaned_text"] for chunk in chunks]

    # Run topic modeling on all texts using "all" model
    topics_all, _ = topic_models["all"].transform(all_cleaned_texts)

    for idx, chunk in enumerate(chunks):
        all_topic_id = topics_all[idx]
        if all_topic_id == -1:
            continue  # Ignore topic -1

        # Get topic label and keywords from "all" model
        all_topic_label, all_topic_keywords = find_topic_label(all_topic_id, "all")

        processed_chunks_all.append({
            "pdf_id": chunk["pdf_id"],
            "category": "all",  # Assigned to "all" category
            "title": chunk["title"],
            "processed_text": chunk["processed_text"],
            "link": chunk["link"],
            "date": chunk["date"],
            "cleaned_text": chunk["cleaned_text"],
            "topic_label": all_topic_label,
            "topic_keywords": all_topic_keywords
        })

    return processed_chunks_all


def process_category_chunks(chunks, topic_models, topic_keywords_mappings):
    """
    Process chunks based on their specific category models efficiently.
    - Groups chunks by category.
    - Loads each category model once.
    - Processes all chunks in that category in one batch.
    """
    processed_chunks_category = []

    # Group chunks by category
    category_chunks = defaultdict(list)
    for chunk in chunks:
        category = chunk.get("category")
        if category in topic_models:  # Ensure category exists in models
            category_chunks[category].append(chunk)

    # Process chunks category-wise
    for category, chunk_list in category_chunks.items():
        topic_model = topic_models[category]  # Load model once

        cleaned_texts = [chunk["cleaned_text"] for chunk in chunk_list]
        topics_category, _ = topic_model.transform(cleaned_texts)  # Batch process

        for idx, chunk in enumerate(chunk_list):
            category_topic_id = topics_category[idx]

            if category_topic_id == -1:
                continue  # Ignore topic -1

            # Get topic label and keywords from category model
            category_topic_label, category_topic_keywords = find_topic_label(category_topic_id, category)

            processed_chunks_category.append({
                "pdf_id": chunk["pdf_id"],
                "category": category,  # Assigned to its specific category
                "title": chunk["title"],
                "processed_text": chunk["processed_text"],
                "link": chunk["link"],
                "date": chunk["date"],
                "cleaned_text": chunk["cleaned_text"],
                "topic_label": category_topic_label,
                "topic_keywords": category_topic_keywords
            })

    return processed_chunks_category


def main(base_dir):
    """
    Main function to process data and generate a final structured JSON output.
    """
    # Define categories
    categories = {
        "all": "model_info_all",
        "business": "model_info_business",
        "education": "model_info_education",
        "government": "model_info_government",
        "others": "model_info_others",
    }

    # Load BERTopic models
    topic_models, topic_keywords_mappings = load_models(base_dir, categories)

    # Load processed chunks with cleaned text
    input_json_path = os.path.join(base_dir, "data/processed_chunks_with_cleaned_text.json")
    with open(input_json_path, "r", encoding="utf-8") as f:
        processed_chunks = json.load(f)

    # Step 1: Process "All" model first
    processed_chunks_all = process_all_chunks(processed_chunks, topic_models, topic_keywords_mappings)

    # Step 2: Process category-specific models
    processed_chunks_category = process_category_chunks(processed_chunks, topic_models, topic_keywords_mappings)

    # Combine
    all_structured_data = processed_chunks_all + processed_chunks_category

    # Check for missing fields before saving
    check_missing_fields(all_structured_data)

    # Save the final dataset
    output_json_path = os.path.join(base_dir, "filterable_database.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_structured_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Successfully saved {len(all_structured_data)} chunks with topics to: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document chunks and generate topic-labeled JSON.")
    parser.add_argument("base_dir", type=str, help="Base directory containing model files and input data.")
    args = parser.parse_args()

    main(args.base_dir)
