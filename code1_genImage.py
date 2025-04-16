!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:47:27 2025

@author: callerw

The following is about genImage by using the detected image.
First load the image then use api to call OCR, then call my LLaVA model (local run), after use api to gen image.
"""

import requests
import json
import os
import time
import re
from PIL import Image
from io import BytesIO
import ollama


def extract_text_from_image(image_path, api_key='xxxxxxx', language='auto'):
    """Extract text from an image using OCR.space API"""
    try:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': language,
            'ocrengine': 2
        }

        with open(image_path, 'rb') as f:
            r = requests.post(
                'https://api.ocr.space/parse/image',
                files={image_path: f},
                data=payload,
            )

        result = r.content.decode()
        json_data = json.loads(result)

        text = ""
        if json_data.get("ParsedResults"):
            text = json_data["ParsedResults"][0]["ParsedText"]

        print(f"Extracted text: {text}")
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def identify_items_with_llava(image_path, ocr_text):
    """Use LLaVA to identify consumer items"""
    try:
        prompt = f"Follow these steps to analyse the receipt image: Step 1: Carefully analyse the extracted OCR text: '{ocr_text}'. Step 2: List out all purchased items you can identify from the OCR text, including food, clothing, electronics, or any other consumer products. Step 3: Translate all identified items into English if they appear in another language (such as Chinese). Step 4: Convert each identified product into a single, concise English word (e.g., 'apple' instead of 'Fuji apple 500g'). Your final response must be in English as a simple comma-separated list of single words representing each product, without any introduction, explanation, or categorisation."

        response = ollama.chat(
            model="llava:7b",
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ]
        )

        extracted_items = response['message']['content']
        print(f"Identified items: {extracted_items}")
        return extracted_items
    except Exception as e:
        print(f"Error identifying items with LLaVA: {e}")
        return ocr_text


def process_identified_items(items_text):
    """Process the LLaVA output to get a clean list of items"""
    # Remove introductory phrases and numbering
    cleaned_text = re.sub(
        r'(here are|i can see|i see|these are|the objects|in the image|item\s*\d+).*?[:.]',
        '',
        items_text,
        flags=re.IGNORECASE
    )

    # Remove any remaining numbers and special characters
    cleaned_text = re.sub(r'[\d•·、，（）()]', ' ', cleaned_text)

    # Split by common separators and clean
    items = re.split(r'[,\n\-–—]', cleaned_text)
    items = [re.sub(r'^\W+|\W+$', '', item).strip().lower() for item in items]

    # Filter valid single-word English items
    valid_items = [
        item for item in items if item and re.match(r'^[a-z]+$', item)]

    return valid_items


def generate_image(prompt, filename="generated_image.png", api_key="xxxxxx"):
    """Generate an image using the Stable Horde API"""
    GENERATE_URL = "https://stablehorde.net/api/v2/generate/async"

    payload = {
        "prompt": prompt,
        "negative_prompt": "details, texture, shading, crosshatching, stippling, multiple lines, noise, grainy",
        "params": {
            "width": 320,
            "height": 320,
            "steps": 20,
            "cfg_scale": 8,
            "sampler_name": "k_euler_a"
        },
        "nsfw": False,
        "models": ["Deliberate"],
        "r2": True,
        "n": 1
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": api_key
    }

    try:
        response = requests.post(
            GENERATE_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 202:
            print(f"Error: {response.status_code}")
            return None

        job_id = response.json().get("id")
        check_url = f"https://stablehorde.net/api/v2/generate/check/{job_id}"
        start_time = time.time()

        while time.time() - start_time < 30:
            time.sleep(1)
            check_response = requests.get(check_url)
            status = check_response.json()
            if status.get("done"):
                break

        status_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
        status_response = requests.get(status_url)
        result = status_response.json()

        if result.get("generations"):
            image_url = result["generations"][0]["img"]
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save(filename)
            return filename
        return None

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


if __name__ == "__main__":
    base_prompt = "single continuous line drawing of a {object} outline, minimalist, black line on white background, ultra clean, no details inside, pure outline only, vectorized look, logo-like, monochrome"
    image_path = "paper_detection.jpg"

    if os.path.exists(image_path):
        ocr_text = extract_text_from_image(image_path)
        identified_items_text = identify_items_with_llava(image_path, ocr_text)
        identified_items = process_identified_items(identified_items_text)
    else:
        identified_items = []

    # Priority selection: 4 > 3 > 2 > 1 > apple
    priority_order = [3, 2, 1, 0]  # Indices for items 4-3-2-1
    selected_item = next(
        (identified_items[i]
         for i in priority_order if i < len(identified_items)),
        "apple"
    )

    prompt = base_prompt.format(object=selected_item)
    output_filename = f"{selected_item.replace(' ', '_')}_outline.png"

    if generate_image(prompt, output_filename):
        print(f"Successfully generated {output_filename}")
    else:
        print("Image generation failed")
