# genai-interactive-topics

## Requirements

To set up the environment and install all dependencies, run:

```bash
pip install -r requirements.txt
```

## How to run

replace `path_to_your_data_directory` and `your_openai_key` with the actual paths and keys you are using.

To run both topic modeling and the app:
```bash
python main.py --mode both --data_dir "path_to_your_data_directory" --openai_key "your_openai_key"
```

To run only the topic modeling:
```bash
python main.py --mode modeling --data_dir "path_to_your_data_directory" --openai_key "your_openai_key"
```

To run only the app:
```bash
python main.py --mode app --data_dir "path_to_your_data_directory"
```
