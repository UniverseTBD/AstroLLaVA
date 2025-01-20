#!/usr/bin/env python3

import argparse
import requests
import urllib3
import json
import concurrent.futures
import time

allowed_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'gif']

def extension(url):
    return url.split('.')[-1]

def download_url(item):
    date, url = item["date"], item["hdurl"]
    ext = extension(url)
    #print(f"Downloading {date}: {url}")
    if not ext in allowed_extensions:
        print('Skipping date:', date, url)
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"images/{date}.{ext}", "wb") as file:
            file.write(response.content)
        print(f"Downloaded {date}: {url}")
        time.sleep(1) # give them a breather
    else:
        print(f"Failed to download {date}: {url}")
        time.sleep(1) # give them a breather

def main():
    parser = argparse.ArgumentParser(description="Download URLs from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing the data.")
    args = parser.parse_args()

    with open(args.json_file) as file:
        print(f'Data loaded from {args.json_file}')
        data = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(download_url, data)

if __name__ == "__main__":
    main()
