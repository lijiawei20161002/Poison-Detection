from dotenv import load_dotenv
import os
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Log in to Hugging Face Hub
if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub.")
else:
    print("HF_TOKEN not found in environment variables. Please set it in your .env file or as an environment variable.")