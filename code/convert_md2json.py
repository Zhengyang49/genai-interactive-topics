import os
import re
import json
import logging
import argparse
import nltk
from nltk.tokenize import word_tokenize
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema import Document

# Ensure that nltk data is downloaded
nltk.download('punkt')

# Define logger
logging.basicConfig(level=logging.INFO)

def split_markdown_by_section(markdown_text):
    """Splits the Markdown text into sections based on headers and formatting."""
    markdown_text = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_text)  # Remove images
    markdown_text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", markdown_text)  # Remove emails
    markdown_text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown_text)  # Remove links
    markdown_text = re.sub(r"\b(?:https?://)(?:\S|\s)+?(?=\s|$)", "", markdown_text)  # Remove URLs
    markdown_text = re.sub(r"\b(?:www\.)\S+\b", "", markdown_text)  # Remove www links
    
    sections = re.split(r"(?<=\n)(#{1,6} .+?)(?=\n)|(?<=\n)(\*\*[A-Z0-9].+?\*\*)(?=\n)", markdown_text)
    sections = [section.strip() for section in sections if section]
    
    result = []
    current_section = ""
    for section in sections:
        if re.match(r"^(#{1,6} .+?|\*\*[A-Z0-9].+?\*\*)$", section):
            if current_section:
                result.append(current_section)
            current_section = section
        else:
            current_section += "\n\n" + section
    
    if current_section:
        result.append(current_section)
    
    return result

def filter_sections(sections):
    """Filter out unwanted sections based on headers or length."""
    return [section for section in sections if len(section.split()) >= 50]

def process_markdown_folder(input_dir, output_file):
    """Process all Markdown files and save results as JSON."""
    all_sections = []
    markdown_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".md")]
    
    token_splitter = NLTKTextSplitter(chunk_size=5000)
    for markdown_file in markdown_files:
        try:
            with open(markdown_file, "r", encoding="utf-8") as file:
                markdown_text = file.read()
                sections = split_markdown_by_section(markdown_text)
                filtered_sections = filter_sections(sections)
                
                for section in filtered_sections:
                    chunks = token_splitter.split_text(section)
                    for chunk in chunks:
                        doc = Document(page_content=chunk, metadata={"source": markdown_file})
                        all_sections.append(doc.page_content)
        except Exception as e:
            logging.error(f"Error processing file {markdown_file}: {e}")
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_sections, file, ensure_ascii=False, indent=4)
    logging.info(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Markdown files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()
    
    process_markdown_folder(args.input_dir, args.output_file)
