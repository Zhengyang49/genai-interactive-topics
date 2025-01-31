import os
import re
import json
import yaml
import openai
import logging
import time
import spacy
import numpy as np
import nltk
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
nlp.max_length = 2800000
logger = logging.getLogger(__name__)

# Helper Functions
def remove_ner(text):
    doc = nlp(text)
    lemmatized_tokens = []
    removed_entities = {}

    for token in doc:
        if token.ent_type_ in entity_labels:
            if token.ent_type_ not in removed_entities:
                removed_entities[token.ent_type_] = []
            removed_entities[token.ent_type_].append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)

    lemmatized_text = ' '.join(token.lemma_ for token in doc if token.ent_type_ not in entity_labels)
    return lemmatized_text.strip(), removed_entities

custom_stopwords = {'references', 'appendix', 'glossary', 'table of contents', 'acknowledgments',
                    'disclosure statement', 'author', 'contact', 'executive summary', 'introduction',
                    'funding', 'citation', 'endnotes', 'notes', 'abstract', 'bibliography'}

def preprocess_text(text):
    text = text.lower()
    processed_text, _ = remove_ner(text)
    processed_text = re.sub(r"####|[*]{2}|[-]{3,}", "", processed_text)
    processed_text = re.sub(r"\b\d+\b", "", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()
    processed_text = re.sub(r'_', ' ', processed_text)
    words = word_tokenize(processed_text)
    words = [word for word in words if word.isalpha() and word not in stop_words and word not in custom_stopwords]
    return " ".join(words)

def preprocess_chunks(chunks):
    return [preprocess_text(chunk) for chunk in chunks]

def load_bertopic_model(model_dir):
    model_path = os.path.join(model_dir, 'topic_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return BERTopic.load(model_path)

def load_json_files(model_dir):
    keywords_path = os.path.join(model_dir, 'topic_keywords_with_labels_gpt.json')
    sizes_path = os.path.join(model_dir, 'topic_sizes_with_labels_gpt.json')

    if not os.path.exists(keywords_path):
        raise FileNotFoundError(f"Keywords file not found at: {keywords_path}")
    if not os.path.exists(sizes_path):
        raise FileNotFoundError(f"Sizes file not found at: {sizes_path}")

    with open(keywords_path, 'r') as f:
        topic_keywords_with_labels = json.load(f)
    with open(sizes_path, 'r') as f:
        topic_sizes = json.load(f)

    return topic_keywords_with_labels, topic_sizes

def load_cluster_descriptions(model_dir):
    des_path = os.path.join(model_dir, 'cluster_descriptions_gpt.json')
    if not os.path.exists(des_path):
        raise FileNotFoundError(f"Cluster descriptions file not found at: {des_path}")
    with open(des_path, 'r') as f:
        cluster_descriptions = json.load(f)
    return cluster_descriptions

# Function to save cluster_descriptions into a JSON file
def save_cluster_descriptions_to_json(model_dir, cluster_descriptions, output_filename="cluster_descriptions.json"):
    # Convert keys to standard int type
    cluster_descriptions = {int(key): value for key, value in cluster_descriptions.items()}
    # Save to JSON
    with open(os.path.join(model_dir, output_filename), "w", encoding="utf-8") as file:
        json.dump(cluster_descriptions, file, ensure_ascii=False, indent=4)
    print(f"Cluster descriptions saved to {output_filename}")

# Function to cluster topics based on intertopic distances
def cluster_topics(topic_model, coords, random_state=42):
    # Set random seed for reproducibility
    np.random.seed(random_state)
    x_vals = coords['x']
    y_vals = coords['y']

    # Standardize the coordinates
    coordinates = np.vstack((x_vals, y_vals)).T
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coordinates)

    # Perform hierarchical clustering
    linkage_matrix = linkage(scaled_coords, method='ward')
    clusters = fcluster(linkage_matrix, t=1, criterion='distance')
    return clusters

# Function to order clusters by importance
def order_clusters_by_importance(topic_model, clusters, coords, random_state=42):
    # Set random seed for reproducibility
    np.random.seed(random_state)
    x_vals = coords['x']
    y_vals = coords['y']
    coordinates = np.vstack((x_vals, y_vals)).T

    # Standardize the coordinates
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coordinates)

    # Apply PCA to extract variance contributions
    pca = PCA(n_components=2, random_state=random_state)  # Use 2 components to match x, y dimensions
    pca_coords = pca.fit_transform(scaled_coords)
    explained_variance = pca.explained_variance_ratio_
    
    # Assign PCA contributions to clusters
    cluster_variance = {}
    for cluster_id in np.unique(clusters):
        cluster_points = pca_coords[np.array(clusters) == cluster_id]
        # Sum variance contributions of all points in the cluster
        cluster_variance[cluster_id] = np.sum(np.var(cluster_points, axis=0) * explained_variance)
    # Order clusters by variance contribution
    ordered_clusters = sorted(cluster_variance.items(), key=lambda x: x[1], reverse=True)
    return ordered_clusters

# Function to display top keywords for the selected topic
def display_keywords(topic_label, topic_keywords_with_labels):
    if topic_label in topic_keywords_with_labels:
        keywords = topic_keywords_with_labels[topic_label]
        return [html.H4(f"Top Keywords for {topic_label}", className="card-title"),
                html.P(f"{', '.join(keywords)}", className="card-text", style={"font-size": "16px"})]
    return html.P("No keywords available.", style={"font-size": "16px", "color": "black"})

def main(directory, rerun=False, key=None):
    # Load the BERTopic model and files   
    model_dir = directory
    openai.api_key = key
    topic_model = load_bertopic_model(model_dir)
    topic_keywords_with_labels, topic_sizes = load_json_files(model_dir)
    coords = np.load(model_dir+"/intertopic_coords.npz")

    # Mapping of topic labels to topic IDs
    label_to_id = {label: topic_id for topic_id, label in enumerate(topic_keywords_with_labels.keys())}
    topic_labels = list(topic_keywords_with_labels.keys())

    # Generate Clustering 
    clusters = cluster_topics(topic_model, coords, random_state=42)
    ordered_clusters = order_clusters_by_importance(topic_model, clusters, coords, random_state=42)
    # Create a mapping from the original cluster IDs to new ordered IDs
    cluster_reorder_map = {old_id: new_id for new_id, (old_id, _) in enumerate(ordered_clusters, start=1)}
    # Apply the mapping to reorder the cluster labels
    reordered_clusters = np.array([cluster_reorder_map[c] for c in clusters])

    if rerun == True:
        if openai.api_key != None:
            cluster_descriptions = generate_cluster_descriptions_GPT(reordered_clusters, topic_keywords_with_labels)
            # Call the function to save the descriptions
            save_cluster_descriptions_to_json(model_dir, cluster_descriptions, "cluster_descriptions_GPT.json")
        else:
            cluster_descriptions = generate_cluster_descriptions(reordered_clusters, topic_keywords_with_labels)
            save_cluster_descriptions_to_json(model_dir, cluster_descriptions, "cluster_descriptions.json")
    else:
        cluster_descriptions = load_cluster_descriptions(model_dir)
    
    # Initialize the Dash app with a Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    def create_intertopic_distance_plot(coords, highlighted_label=None):
        x_vals = coords['x']
        y_vals = coords['y']
    
        # Add jitter to reduce overlap
        def add_jitter(values, jitter_strength=0.5):
            np.random.seed(4)
            return values + np.random.uniform(-jitter_strength, jitter_strength, size=len(values),)
    
        x_vals = add_jitter(x_vals)
        y_vals = add_jitter(y_vals)
    
        hover_texts = [
            f"Topic: {label}<br>Documents: {topic_sizes.get(label,0)}<br>Keywords: {', '.join(topic_keywords_with_labels[label])}"
            for label in topic_labels]
        marker_colors = ['red' if label == highlighted_label else 'lightblue' for label in topic_labels]
    
        # Normalize and invert the marker sizes
        max_size = max(topic_sizes.values())
        min_size = min(topic_sizes.values())
    
        scale_factor = 0.7
        marker_sizes = [
            15 + (size - min_size) ** scale_factor / (max_size - min_size) ** scale_factor * 20
            for size in topic_sizes.values()]
    
        scatter_trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            text=topic_labels,
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=1.5, color='black')),
            textposition='bottom center',
            textfont=dict(size=12, color='black'))
    
        fig = go.Figure(data=[scatter_trace])
        fig.update_layout(
            title="Intertopic Distance Map",
            height=700,
            xaxis_title="PC1",
            yaxis_title="PC2",
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode="closest",
            font=dict(family="Arial", size=12, color="black"),
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        return fig

    # Function to create a bar chart of top keywords and their weights for a specific topic
    def visualize_topic_term_single(topic_model, topic_label, topic_keywords_with_labels, n_words=10) -> go.Figure:
        if topic_label in topic_keywords_with_labels:
            topic_id = label_to_id.get(topic_label)
            if topic_id is not None:
                words_and_scores = topic_model.get_topic(topic_id)[:n_words]
                words = [word for word, _ in words_and_scores][::-1]  # Reverse order for display
                scores = [score for _, score in words_and_scores][::-1]  # Reverse order for display
    
                fig = go.Figure(go.Bar(x=scores, y=words, orientation="h", marker_color="#636EFA",))
                fig.update_layout(
                    title=f"Top {n_words} Keywords for Topic {topic_label}",
                    xaxis_title="Weight",
                    yaxis_title="Keywords",
                    height=500,
                    template="plotly_white")
                return fig
        return go.Figure()
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Interactive Topic Model", className="display-4"),
                html.Hr(),
                html.P("Select a topic using the buttons or the dropdown below:", className="lead"),
                html.Button('Previous Topic', id='prev-topic', n_clicks=0),
                html.Button('Next Topic', id='next-topic', n_clicks=0),
                html.Button('Clear Topic', id='clear-topic', n_clicks=0),
                dcc.RadioItems(
                    id="topic-selector",
                    options=[{'label': label, 'value': label} for label in topic_keywords_with_labels.keys()],
                    labelStyle={'display': 'block'},
                    style={"height": "300px", "overflow-y": "scroll"},
                ),
                dbc.Button("Reset View", id="reset-button", color="primary", className="mt-3"),
            ], width=3),
    
            dbc.Col([
                dcc.Graph(id='intertopic-plot', figure=create_intertopic_distance_plot(coords)),
            ], width=9)
        ], className="mb-4"),
    
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id='keyword-display')
                    ])
                ], style={"height": "100%", "background-color": "#e3f2fd"})
            ], width=6),
    
            dbc.Col([
                dcc.Graph(id="term-bar-chart", style={"height": "500px"})
            ], width=6)
    
        ]),
    
        dbc.Row([
            dbc.Col([
                html.Div(id='cluster-descriptions', style={"margin-top": "20px", "padding": "10px", "background-color": "#f0f8ff", "border-radius": "10px"})
            ])
        ])
    ], fluid=True)

    # Callbacks for interactivity
    @app.callback(
        [Output('intertopic-plot', 'figure'),
         Output('keyword-display', 'children'),
         Output('term-bar-chart', 'figure')],
        [Input('topic-selector', 'value'),
         Input('intertopic-plot', 'clickData'),
         Input('reset-button', 'n_clicks')],
        [State('topic-selector', 'value')])
    def update_plot_and_display(selected_label, click_data, reset_clicks, current_label):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id']
    
        if triggered == 'reset-button.n_clicks':
            # Reset view and clear selections
            return create_intertopic_distance_plot(coords), \
                   html.H5("Select a topic to see the top keywords.", style={"font-size": "18px", "color": "black"}), \
                   go.Figure()
    
        if triggered == 'intertopic-plot.clickData':
            # Handle node click
            clicked_label = click_data['points'][0]['text']
            selected_label = clicked_label
    
        if selected_label:
            # Highlight selected topic
            intertopic_plot = create_intertopic_distance_plot(coords, highlighted_label=selected_label)
            keywords_display = display_keywords(selected_label, topic_keywords_with_labels)
            term_bar_chart = visualize_topic_term_single(topic_model, selected_label, topic_keywords_with_labels)
            return intertopic_plot, keywords_display, term_bar_chart
    
        return create_intertopic_distance_plot(coords), \
               html.H5("Select a topic to see the top keywords.", style={"font-size": "18px", "color": "black"}), \
               go.Figure()

    # Adjust overlapping labels
    def adjust_label_positions(x_vals, y_vals, labels):
        jitter_strength = 0.03
        adjusted_x = x_vals + np.random.uniform(-jitter_strength, jitter_strength, len(x_vals))
        adjusted_y = y_vals + np.random.uniform(-jitter_strength, jitter_strength, len(y_vals))
        return adjusted_x, adjusted_y

    # Update cluster descriptions
    @app.callback(
        Output('cluster-descriptions', 'children'),
        Input('reset-button', 'n_clicks')
    )
    def update_cluster_descriptions(n_clicks):
        descriptions = [html.P(description) for description in cluster_descriptions.values()]
        return descriptions
    
    # Navigation buttons to cycle through topics
    @app.callback(
        Output('topic-selector', 'value'),
        [Input('prev-topic', 'n_clicks'), Input('next-topic', 'n_clicks'), Input('clear-topic', 'n_clicks')],
        [State('topic-selector', 'value')]
    )
    def navigate_topics(prev_clicks, next_clicks, clear_clicks, current_value):
        current_index = topic_labels.index(current_value) if current_value else 0
        if clear_clicks > 0:
            return None
        elif prev_clicks > 0 and current_index > 0:
            return topic_labels[current_index - 1]
        elif next_clicks > 0 and current_index < len(topic_labels) - 1:
            return topic_labels[current_index + 1]
        return current_value
    
    def reset_view(n_clicks):
        if n_clicks:
            return None  # Reset topic selection to None

    # Run the app
    app.run_server(debug=True, port=8092)



if __name__ == '__main__':
    openai_key = None
    trigger = False
    main(data_dir, rerun=False, key=openai_key)
