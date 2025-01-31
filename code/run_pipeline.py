import os
import argparse
from convert_md2json import process_markdown_folder
from process_text import preprocess_chunks
from topic_modeling_json import main as topic_modeling_main
from visualization import main as visualization_main

def run_pipeline(input_dir, output_dir, config_file, openai_api_key=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Convert Markdown to JSON
    json_path = os.path.join(output_dir, "data.json")
    process_markdown_folder(input_dir, json_path)
    
    # Step 2: Preprocess Data
    processed_data_path = os.path.join(output_dir, "processed_data.json")
    preprocess_chunks(json_path, processed_data_path)
    
    # Step 3: Run Topic Modeling
    topic_modeling_main(processed_data_path, config_file, output_dir, openai_api_key)
    
    # Step 4: Run Visualization
    visualization_main(output_dir)
    
    print(f"Pipeline completed. All outputs are stored in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Markdown files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data and model")
    parser.add_argument("--config_file", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--openai_api_key", type=str, required=False, help="OpenAI API Key for GPT-based labeling")
    args = parser.parse_args()
    
    run_pipeline(args.input_dir, args.output_dir, args.config_file, args.openai_api_key)
