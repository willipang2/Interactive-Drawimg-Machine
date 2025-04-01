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


def extract_text_from_image(image_path, api_key='xxxxxx', language='auto'):
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

        # Extract the parsed text
        text = ""
        if json_data.get("ParsedResults"):
            text = json_data["ParsedResults"][0]["ParsedText"]

        print(f"Extracted text: {text}")
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def identify_items_with_llava(image_path, ocr_text):
    """Use LLaVA to identify consumer items from the image and OCR text"""
    try:
        prompt = f"Look at this receipt image, which can be in Chinese or English. The OCR extracted the following text: '{ocr_text}'. List all the purchased items such as food, clothing items, books, toys, or other consumer products shown in the image or mentioned in the text. Format your response must in English as a simple comma-separated list without any introduction, explanation, or categorization."

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
        return ocr_text  # Fallback to OCR text if LLaVA fails


def process_identified_items(items_text):
    """Process the LLaVA output to get a clean list of items"""
    cleaned_text = re.sub(
        r'(here are|i can see|i see|these are|the objects|in the image).*?:', '', items_text, flags=re.IGNORECASE)

    # Split by common separators
    items = re.split(r'[,\n•-]', cleaned_text)
    items = [item.strip() for item in items if item.strip()]

    return items


def generate_image(prompt, filename="generated_image.png", api_key="xxxxx"):
    """Generate an image using the Stable Horde API"""
    # API endpoints
    GENERATE_URL = "https://stablehorde.net/api/v2/generate/async"

    payload = {
        "prompt": prompt,
        "negative_prompt": "details, texture, shading, crosshatching, stippling, multiple lines, noise, grainy",
        "params": {
            "width": 320,
            "height": 320,
            "steps": 20,
            "cfg_scale": 8,
            "sampler_name": "k_euler_a"  # Fast sampler
        },
        "nsfw": False,
        "models": ["Deliberate"],
        "r2": True,
        "n": 1
    }

    # Headers with API key
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key
    }

    print(f"Submitting request with prompt: '{prompt}'")

    try:
        response = requests.post(
            GENERATE_URL, headers=headers, data=json.dumps(payload))

        if response.status_code != 202:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

        # Get the job ID
        job_id = response.json().get("id")
        print(f"Request submitted! Job ID: {job_id}")

        check_url = f"https://stablehorde.net/api/v2/generate/check/{job_id}"
        max_wait_time = 30  # Maximum wait time in seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            time.sleep(1)
            check_response = requests.get(check_url)
            status = check_response.json()
            print(
                f"Status: {status.get('waiting')} waiting, {status.get('processing')} processing, {status.get('finished')}/{status.get('amount')} finished")

            if status.get("done"):
                break

        status_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
        status_response = requests.get(status_url)
        result = status_response.json()

        if result.get("generations") and len(result["generations"]) > 0:
            image_url = result["generations"][0]["img"]
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img.save(filename)
            print(f"Image saved as {filename}")
            return filename
        else:
            print("No generations were returned.")
            return None

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


# Main function
if __name__ == "__main__":
    #  prompt template
    base_prompt = "single continuous line drawing of a {object} outline, minimalist, black line on white background, ultra clean, no details inside, pure outline only, vectorized look, logo-like, monochrome"
    image_path = "paper_detection.jpg"

    if os.path.exists(image_path):
        # Step 1: Extract text from the image using OCR
        ocr_text = extract_text_from_image(image_path)

        # Step 2: Use LLaVA to identify consumer items from the image and OCR text
        identified_items_text = identify_items_with_llava(image_path, ocr_text)

        # Step 3: Process the LLaVA output to get a clean list of items
        identified_items = process_identified_items(identified_items_text)

        if not identified_items:
            print("No items identified. Using default object.")
            identified_items = ["book"]
    else:
        print(f"Image '{image_path}' not found. Using default object.")
        identified_items = ["book"]

    # Step 4: Generate only one image using the first identified item
    if identified_items:
        item = identified_items[0]  # now set as the first item
        prompt = base_prompt.format(object=item)
        output_filename = f"{item.replace(' ', '_')}_outline.png"
        result = generate_image(prompt, output_filename)

        if result:
            print(f"Generated image: {result}")
        else:
            print(f"Failed to generate image for {item}")
