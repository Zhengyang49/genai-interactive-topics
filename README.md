# Topic Modeling Pipeline

## Overview

This repository contains a pipeline for topic modeling using BERTopic. The workflow involves:

1. **First-time setup**: Convert markdown data to JSON, preprocess the data, run topic modeling, generate topic descriptions, and visualize topics interactively.
2. **Using an existing model**: Load the saved model and descriptions to generate interactive visualizations.

## Installation & Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd <repository_name>
```
### 2. Requirements
To set up the environment and install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

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

## How to deploy
We include two types of publishing the app to a real website: one is through using google cloud platform, one is through our own server.

### Running on our own server
#### 1. Open a port (e.g., 8050):
```
sudo ufw allow 8050
```

#### 2. Run the app
Then you can access the application through the server's IP and port (e.g., http://your-server-ip:8050).
```
python app.py
```

#### 3. Configuring Nginx for HTTP/HTTPS Access
If you want to access your Dash application through standard HTTP/HTTPS, you can set up a reverse proxy using Nginx.

1. Install Nginx

    Use the following commands on your server to install nginx:

        sudo apt update
        sudo apt install nginx
    
2. Configure Nginx Reverse Proxy:

    1). Open a new Nginx configuration file
    ```
    sudo nano /etc/nginx/sites-available/dash-app
    ```
    2). Add the following configuration (replace your-domain.com and 127.0.0.1:8050 with your domain and Dash app's local address, respectively)
    ```
    server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
    ```
    3). Save and close the file.

    4). Enable the configuration:
    ```
    sudo ln -s /etc/nginx/sites-available/dash-app /etc/nginx/sites-enabled
    sudo systemctl restart nginx
    ```

### Using Google Cloud Platform(GCP)
In this part you can follow the step-by-step instructions to deploy and run our Dash application on Google Cloud Platform (GCP) using **Cloud Run**.

#### 1. Prerequisites
1). Install **Google Cloud SDK** on your local machine.

2). Add **gunicorn** to the requirements.txt

#### 2. Steps to Deploy
1). Set Up GCP Environment

Enable **Cloud Run API** and **Cloud Build API**.

2). Deploy the Application

- Authenticate your local machine with GCP and set the default project:

        gcloud auth login
        gcloud config set project [PROJECT_ID]
- Submit your application for build and deployment:

        gcloud builds submit --tag gcr.io/[PROJECT_ID]/dash-app
- Deploy the application to Cloud Run:

        gcloud run deploy dash-app \
            --image gcr.io/[PROJECT_ID]/dash-app \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated
#### 3. Access the Application
Once deployed, Cloud Run will provide a public URL for your application, such as:

    https://dash-app-xxxxxx.run.app
