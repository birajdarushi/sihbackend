from dotenv import load_dotenv
import os
import json
import csv
from datetime import datetime, timezone, timedelta
import urllib.parse
import asyncio
import aiohttp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm.asyncio import tqdm as async_tqdm
import orjson
from langdetect import detect_langs
from googletrans import Translator
from bs4 import BeautifulSoup
import torch
from transformers import pipeline
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError

# Explicitly set environment variables for CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

load_dotenv()

# Load Azure Blob Storage credentials from environment variables
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Create ContainerClient
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

# Create the container if it doesn't exist
try:
    container_client.create_container()
    print(f"Container '{AZURE_CONTAINER_NAME}' created.")
except ResourceExistsError:
    print(f"Container '{AZURE_CONTAINER_NAME}' already exists.")

# GPU and Device Configuration
def get_device():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("CUDA is not available. Falling back to CPU.")
        return torch.device('cpu')

# Set device globally
device = get_device()

# Additional Troubleshooting Function
def check_cuda_environment():
    print("CUDA Availability Check:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print("CUDA Device Details:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Call environment check at the start
check_cuda_environment()

# Initialize models with explicit device handling
translator = Translator()
ner_model = pipeline("ner", 
                     model="dbmdz/bert-large-cased-finetuned-conll03-english", 
                     device=0 if torch.cuda.is_available() else -1)
classification_model = pipeline("text-classification", 
                                model="distilbert-base-uncased", 
                                device=0 if torch.cuda.is_available() else -1)
summarization_model = pipeline("summarization", 
                               model="facebook/bart-large-cnn", 
                               device=0 if torch.cuda.is_available() else -1)

# Disaster classification label mapping
label_to_disaster = {
    'LABEL_0': "earthquake",
    'LABEL_1': "flood",
    'LABEL_2': "fire",
    'LABEL_3': "landslide",
    'LABEL_4': "wildfire",
    'LABEL_5': "windstorm",
    'LABEL_6': "drought",
    'LABEL_7': "pests",
}

# Caching for API responses
cache = {}

@lru_cache(maxsize=1000)
def translate_to_english(text):
    try:
        detected_langs = detect_langs(text)
        if detected_langs[0].lang != 'en' and detected_langs[0].prob > 0.9:
            translated = translator.translate(text, dest='en')
            return translated.text
        else:
            return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def classify_disaster(text):
    classification = classification_model(text, top_k=1)[0]
    label_id = classification['label']
    predicted_disaster = label_to_disaster.get(label_id, "Unknown")

    disaster_keywords = {
        "earthquake": ["earthquake", "tremor", "quake"],
        "flood": ["flood", "inundation", "deluge"],
        "fire": ["fire", "wildfire", "blaze", "burning"],
        "landslide": ["landslide", "mudslide"],
        "wildfire": ["wildfire", "forest fire"],
        "windstorm": ["windstorm", "hurricane", "cyclone", "typhoon", "storm"],
        "drought": ["drought", "dry spell"],
        "pests": ["locust", "pest", "infestation"]
    }

    text_lower = text.lower()
    for disaster_type, keywords in disaster_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return disaster_type

    return predicted_disaster

def extract_location(text):
    try:
        entities = ner_model(text)
        locations = set()
        for entity in entities:
            if entity['entity'] == 'I-LOC':
                if len(locations) > 0 and entity['word'].startswith('##'):
                    last_location = list(locations)[-1]
                    locations.remove(last_location)
                    locations.add(last_location + entity['word'][2:])
                else:
                    locations.add(entity['word'])
        return ', '.join(locations)
    except Exception as e:
        print(f"Location extraction error: {e}")
        return "Unknown Location"

def summarize_text(text):
    try:
        # Calculate lengths based on input text
        input_length = len(text.split())
        max_length = min(input_length - 1, max(10, input_length // 2))
        min_length = max(5, max_length // 2)
        
        summarized = summarization_model(text, 
                                       max_length=max_length, 
                                       min_length=min_length, 
                                       length_penalty=2.0)
        return summarized[0]['summary_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:100] + "..."

async def fetch_twitter_data(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.json()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

async def search_twitter(query, max_tweets):
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "twitter241.p.rapidapi.com"
    }

    tweet_data = []
    count = 20
    remaining_tweets = max_tweets

    async with aiohttp.ClientSession() as session:
        while remaining_tweets > 0:
            encoded_query = urllib.parse.quote(query)
            url = f"https://twitter241.p.rapidapi.com/search-v2?type=Latest&count={count}&query={encoded_query}"
            
            tweets = await fetch_twitter_data(session, url, headers)

            if 'result' in tweets and 'timeline' in tweets['result']:
                timeline = tweets['result']['timeline']
                if 'instructions' in timeline and len(timeline['instructions']) > 0:
                    entries = timeline['instructions'][0]['entries']

                    for entry in entries:
                        content = entry.get('content', {})
                        if content.get('__typename') == 'TimelineTimelineItem':
                            tweet = content.get('itemContent', {}).get('tweet_results', {}).get('result', {})
                            if tweet:
                                user = tweet.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {})
                                username = user.get('screen_name')
                                created_at = tweet.get('legacy', {}).get('created_at')
                                likes = tweet.get('legacy', {}).get('favorite_count', 0)
                                retweets = tweet.get('legacy', {}).get('retweet_count', 0)
                                tweet_text = tweet.get('legacy', {}).get('full_text', '')

                                if created_at:
                                    dt = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
                                    ist_time = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))

                                    tweet_data.append({
                                        "source": "Twitter",
                                        "username": username,
                                        "date": ist_time.strftime('%Y-%m-%d'),
                                        "time": ist_time.strftime('%H:%M:%S'),
                                        "likes": likes,
                                        "retweets": retweets,
                                        "content": tweet_text
                                    })
                                    remaining_tweets -= 1
                                    if remaining_tweets <= 0:
                                        break
                        if remaining_tweets <= 0:
                            break
                if remaining_tweets > 0:
                    continue
                else:
                    break
            else:
                print("No tweets found or incorrect key used.")
                break

    return tweet_data

async def fetch_news_data(session, url, headers, params):
    async with session.get(url, headers=headers, params=params) as response:
        return await response.json()

async def fetch_news():
    url = "https://google-news13.p.rapidapi.com/search"
    querystring = {"lr": "en-IN"}
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "google-news13.p.rapidapi.com"
    }
    async with aiohttp.ClientSession() as session:
        data = await fetch_news_data(session, url, headers, querystring)

    news_data = []

    if 'items' in data and data['items']:
        articles = data['items']
        sorted_articles = sorted(articles, key=lambda x: int(x['timestamp']), reverse=True)
        
        for article in sorted_articles:
            readable_date = datetime.fromtimestamp(int(article['timestamp']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            news_data.append({
                "source": "Google News",
                "title": article['title'],
                "date": readable_date.split()[0],
                "time": readable_date.split()[1],
                "publisher": article['publisher'],
                "url": article['newsUrl'],
                "content": article['snippet'],
                "thumbnail": article['images']['thumbnail']
            })

    return news_data

def upload_to_azure(file_path, blob_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"Uploaded '{file_path}' to Azure Blob Storage as '{blob_name}'.")
    except Exception as e:
        print(f"Failed to upload '{file_path}' to Azure Blob Storage: {e}")

async def process_item(item):
    try:
        translated_content = translate_to_english(item["content"])
        disaster_type = classify_disaster(translated_content)
        location = extract_location(translated_content)
        summary = summarize_text(translated_content)

        processed_item = {
            "source": item["source"],
            "date": item["date"],
            "time": item["time"],
            "content": item["content"],
            "translated_content": translated_content,
            "disaster_type": disaster_type,
            "location": location,
            "summary": summary
        }

        if item["source"] == "Twitter":
            processed_item.update({
                "username": item["username"],
                "likes": item["likes"],
                "retweets": item["retweets"]
            })
        elif item["source"] == "Google News":
            processed_item.update({
                "title": item["title"],
                "publisher": item["publisher"],
                "url": item["url"],
                "thumbnail": item["thumbnail"]
            })

        return processed_item
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

async def process_and_combine_data(twitter_data, news_data):
    combined_data = twitter_data + news_data
    tasks = [process_item(item) for item in combined_data]
    processed_data = await async_tqdm.gather(*tasks, desc="Processing items")

    # Remove None values (failed processing attempts)
    processed_data = [item for item in processed_data if item is not None]

    # Sort combined data by date and time
    processed_data.sort(key=lambda x: (x["date"], x["time"]), reverse=True)

    return processed_data

async def load_disaster_keywords(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

async def prioritize_and_search(keywords_file, max_tweets):
    disaster_data = await load_disaster_keywords(keywords_file)
    
    # Map priorities to a numerical value for sorting
    priority_map = {"high": 1, "medium": 2, "low": 3}
    
    # Sort disasters by priority
    sorted_disasters = sorted(disaster_data.items(), key=lambda x: priority_map[x[1]["priority"]])

    for disaster, details in sorted_disasters:
        # Append " India" after each keyword
        query = " OR ".join(keyword + "" for keyword in details["keywords"])
        output_filename = f"{disaster}.json"
        
        print(f"Processing disaster: {disaster} with query: '{query}'")
        await process_disaster(disaster, query, max_tweets, output_filename)

async def process_disaster(disaster_name, query, max_tweets, output_file):
    try:
        # Fetch Twitter data
        twitter_data = await search_twitter(query, max_tweets)
        
        # Fetch News data
        news_data = await fetch_news()

        # Combine and process data
        combined_data = await process_and_combine_data(twitter_data, news_data)

        # Save results to a file named after the disaster by appending to existing data
        if os.path.exists(output_file):
            with open(output_file, 'rb') as infile:
                existing_data = orjson.loads(infile.read())
            # Append new data to existing data
            combined_data = existing_data + combined_data

        with open(output_file, 'wb') as outfile:
            outfile.write(orjson.dumps(combined_data, option=orjson.OPT_INDENT_2))
        print(f"Results for '{disaster_name}' saved to {output_file}")
        
        # Upload the file to Azure Blob Storage
        upload_to_azure(output_file, output_file)
        
        # Removed the file removal lines to retain the local file
    except Exception as e:
        print(f"Error processing disaster '{disaster_name}': {e}")
async def main():
    start_time = time.time()
    
    # Path to the disaster keywords file
    keywords_file = "disaster_keywords.json"
    max_tweets = 10  # Set a limit for the number of tweets
    
    await prioritize_and_search(keywords_file, max_tweets)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())