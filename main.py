import os
import argparse
import logging
from topic_modeling import run_topic_modeling
from app import run_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Topic Modeling and/or Launch the Interactive App")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["modeling", "app", "both"],
        default="both",
        help="Choose 'modeling' to run topic modeling only, 'app' to run the app only, or 'both' to run both."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="saved_topic_model/",
        help="Directory where the input data and model outputs are stored."
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default=None,
        help="OpenAI API key (if needed for generating descriptions)."
    )
    args = parser.parse_args()

    # Run based on the selected mode
    if args.mode in ["modeling", "both"]:
        logger.info("Starting topic modeling...")
        run_topic_modeling(args.data_dir, args.openai_key)
        logger.info("Topic modeling completed successfully.")

    if args.mode in ["app", "both"]:
        logger.info("Launching the interactive app...")
        run_app(args.data_dir, args.openai_key)
        logger.info("App launched successfully.")

if __name__ == "__main__":
    main()
