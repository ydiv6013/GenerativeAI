import torch,os
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
from dotenv import load_dotenv

# define constants
load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
hf_token = os.getenv('HF_API_KEY')

# Check the API key
if not hf_token:
    print("No API key was found.")
else:
    print("API key found.")

#login to hugging face 
login(hf_token,add_to_git_credential=True)

# Text Generation

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

generator = pipeline("text-generation",device=device)
result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
print(result[0]['generated_text'])