#!/usr/bin/env python3
from openai import OpenAI
import time
import json
import sys
import os

# Initialize the OpenAI client with your API key
client = OpenAI()

# Preparing results files to read
working_dir = os.getcwd()
file_path_origin = os.path.join(working_dir, r'json_formatted_results\origin_nexus_results.json')
file_path_nebula = os.path.join(working_dir, r'json_formatted_results\nebula_nexus_results.json')
file_path_horizon = os.path.join(working_dir, r'json_formatted_results\horizon_nexus_results.json')
# Read the JSON results from the specified full path
with open(file_path_origin, 'r') as file:
    json_results_origin = json.load(file)
with open(file_path_nebula, 'r') as file:
    json_results_nebula = json.load(file)
with open(file_path_horizon, 'r') as file:
    json_results_horizon = json.load(file)

# Combine the JSON results
combined_results = {
    "Origin": json_results_origin,
    "Nebula": json_results_nebula,
    "Horizon": json_results_horizon
}

# Convert the combined JSON object to a string
combined_json_string = json.dumps(combined_results)

print("Generating script...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """ 
You are an AI trained to assist with physical therapy treatment planning. You will create a 
4-week program (4 phases) for the client, considering the data from Origin Nexus, Nebula Nexus, and Horizon Nexus.
These measurements were calculated by a neural network designed by specialists to highlight all the 
weak links in the body. Trust and respect them 100%.    

For each dataset:
- Origin Nexus
- Nebula Nexus
- Horizon Nexus

Review and use the following methods accordingly:

For Release/Lengthen type, use combinations of:
- Self Myofascial Release
- Static stretching
- Dynamic stretching
- Breathing exercises in certain positions to facilitate expansion (decompression) of areas

For Activate/Shorten type, use combinations of:
- Isometric holds
- Resistance training
- Unilateral patterns of exercises

Start gently in the first weeks and deepen as the program progresses.
Focus on multiple muscle (anatomy train lines) activation to use more complex approaches than something isolated.

Before starting to generate, take some time to truly process the data, understanding the biomechanical patterns. 
Offer insights as we go through the weeks.

Don't use "etc.", as the user has to receive some clear indications of exactly what exercises and methods to use 
and why. Use common names so that the user can find out what to do in a google/youtube search.

Make sure you cover all provided muscle groups.
And most importantly, in methods which allow, combine multiple targeted muscle & types into one. 
For example, a modified lateral plank can both activate and stretch many of the targets.
"""
        },
        {
            "role": "user",
            "content": combined_json_string
        }
    ]
)

response_text = response.choices[0].message.content
print(response_text)
