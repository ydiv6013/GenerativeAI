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

"""
#############################################################
sentences = [
    "I am so happy with how things turned out!",
    "This is the worst day of my life.",
    "I can't believe how amazing this is!",
    "I feel so anxious about the upcoming exam.",
    "Everything is so confusing right now.",
    "I am content with what I have achieved.",
    "Winning the award made me feel ecstatic!",
    "I am so bored sitting at home all day.",
    "I get so nervous before giving presentations.",
    "This delay is really frustrating.",
    "I feel so much joy spending time with my family.",
    "I am incredibly grateful for your support.",
    "I am hopeful that things will get better soon.",
    "Why do I always feel so jealous of her success?",
    "I am relieved that everything worked out fine.",
    "I resent being overlooked for the promotion.",
    "That meal was so satisfying!",
    "I feel so much sympathy for the victims of the disaster.",
    "I just don't care anymore; I feel so apathetic.",
    "I've been feeling depressed for weeks now.",
    "Her cheerful attitude always brightens my day.",
    "I feel so calm and at peace here.",
    "I am absolutely elated to have received this honor!",
    "I can't shake off this fear of failure.",
    "I feel so much love for my family.",
    "It's a bittersweet kind of melancholy I feel.",
    "I am highly motivated to achieve my goals.",
    "I am optimistic about the future of our project.",
    "I can't help but feel pessimistic about the outcome.",
    "I feel so proud of everything we've accomplished.",
    "I deeply regret the decisions I've made.",
    "I am so remorseful for hurting your feelings.",
    "That news caught me completely by surprise!",
    "I am so thankful for all the blessings in my life.",
    "I feel so thoughtful and reflective right now.",
    "I trust you with all my heart.",
    "I have been so unhappy since you left.",
    "I am so upset that you didn't tell me earlier.",
    "I feel this overwhelming urge for vengeance.",
    "I can't stop worrying about what might happen next.",
    "I am torn; my feelings are so ambivalent.",
    "His arrogance is starting to bother everyone around him.",
    "I feel so ashamed for what I did.",
    "I am incredibly disappointed in you.",
    "I feel utterly disgusted by his behavior.",
    "The guilt is eating me alive.",
    "I feel humiliated after what happened.",
    "I feel so inspired by her determination.",
    "Life feels so full of hope right now!"
]

# sentiment analysis

classifier = pipeline("sentiment-analysis")
result = classifier("I'm super excited to be on the way to LLM mastery!")
print(result)

for sentence in sentences:
    result = classifier(sentence)
    print(result)
    
#############################################################

# Named Entity Recognisation (NER)

para = "On October 15, 2023, John Smith attended a technology conference hosted by Microsoft at the Moscone Center in San Francisco, California. During the event, he met Dr. Emily Chen, a renowned AI researcher from Stanford University, and discussed the impact of ChatGPT and other OpenAI technologies on modern industries. Later, they attended a keynote session led by Satya Nadella, the CEO of Microsoft, where he announced a new partnership with Tesla to develop innovative solutions for autonomous vehicles. The conference also featured startups like GreenTech Solutions, which unveiled their latest product aimed at reducing carbon emissions. The event concluded with a gala dinner at The Ritz-Carlton, where attendees celebrated the advancements in artificial intelligence and sustainable technologies"
ner = pipeline("ner",grouped_entities=True)
result = ner(para)
print(result)

#############################################################
# Text Summarization

"""
summarizer = pipeline("summarization")
text = """The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
It's an extremely popular library that's widely used by the open-source data science community.
It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
""""""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

"""

#############################################################
# Translation

translator = pipeline("translation_en_to_fr")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print(result[0]['translation_text'])

#############################################################
# Classification

classifier = pipeline("zero-shot-classification") # add device ='cuda' if available
result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
print(result)