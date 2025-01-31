from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import yaml
import openai
import logging
import json

# Function to generate enriched cluster descriptions
def generate_cluster_descriptions(clusters, topic_keywords_with_labels):
    cluster_descriptions = {}
    unique_clusters = sorted(np.unique(clusters)) 

    for cluster_id in unique_clusters:
        # Get topics in this cluster
        topics_in_cluster = [label for label, cluster in zip(topic_keywords_with_labels.keys(), clusters) if cluster == cluster_id]
        keywords = [", ".join(topic_keywords_with_labels[label]) for label in topics_in_cluster]

        # Unique keywords across topics in the cluster
        unique_keywords = set(", ".join(keywords).split(", "))
        
        # Enrich the description
        description = (
            f"Cluster {cluster_id} includes topics: {', '.join(topics_in_cluster)}. "
            f"This cluster focuses on themes revolving around {', '.join(list(unique_keywords)[:5])}, among other concepts. "
            f"The topics in this cluster highlight diverse but interconnected subjects. "
            f"For example, {topics_in_cluster[0]} emphasizes {topic_keywords_with_labels[topics_in_cluster[0]][0]} and "
            f"{topic_keywords_with_labels[topics_in_cluster[0]][1]}, while {topics_in_cluster[-1]} explores aspects such as "
            f"{topic_keywords_with_labels[topics_in_cluster[-1]][-2]} and {topic_keywords_with_labels[topics_in_cluster[-1]][-1]}. "
            "Together, these topics provide a comprehensive overview of the cluster's overarching themes."
            #"indicating a blend of approaches, ranging from theoretical to practical applications."
        )

        # Save the enriched description
        cluster_descriptions[cluster_id] = description

    return cluster_descriptions

# Function to generate enriched cluster descriptions using GPT with retries and exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def generate_description_with_retry(messages):
    # Call the OpenAI ChatCompletion API with retries
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use the appropriate model
        messages=messages,
        max_tokens=400,  # Adjust the token limit as needed
        temperature=0.7  # Control the creativity of the response
    )
    return response

def generate_cluster_descriptions_GPT(clusters, topic_keywords_with_labels):
    cluster_descriptions = {}
    unique_clusters = sorted(np.unique(clusters)) 

    for cluster_id in unique_clusters:
        # Collect topics and keywords for the given cluster
        topics_in_cluster = [label for label, cluster in zip(topic_keywords_with_labels.keys(), clusters) if cluster == cluster_id]
        keywords = [", ".join(topic_keywords_with_labels[label]) for label in topics_in_cluster]

        # Prepare the content for the OpenAI chat model
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

        # Use the function with exponential backoff to generate the description
        try:
            response = generate_description_with_retry(messages)
            description = response.choices[0].message['content'].strip()
        except Exception as e:
            description = f"Error generating description: {str(e)}"

        # Save the generated description
        cluster_descriptions[cluster_id] = f"Cluster {cluster_id} includes topics: {', '.join(topics_in_cluster)}. " + description

    return cluster_descriptions