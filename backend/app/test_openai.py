import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Check if API Key is loaded
if not api_key:
    print("ERROR: OPENAI_API_KEY is not set in the .env file.")
    exit(1)
else:
    print("OPENAI_API_KEY loaded successfully.")

# Initialize OpenAI Client
try:
    client = openai.OpenAI(api_key=api_key)

    # Make a simple OpenAI API call to test if the key works
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "meshing"}]
    )

    # Print AI Response
    print("OpenAI API Response:", response.choices[0].message.content.strip())

except openai.APIError as e:
    print("OpenAI API Error:", str(e))
    exit(1)
