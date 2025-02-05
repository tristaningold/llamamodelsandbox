#remember that you need to run the huggingface-cli login command and then provide and access token
#then you have run the python3 text_to_model.py command

import torch
from llm_output_processor import process_llm_output
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def text_to_model(text):
    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt")

    # Use the Llama model for text analysis
    outputs = model(**inputs)

    # Extract the relevant information from the output
    timelines = []
    applicability = []
    requirements = []

    # Use the output to extract timelines
    timelines_output = outputs.last_hidden_state[:, 0, :]
    timelines.append(timelines_output)

    # Use the output to extract applicability
    applicability_output = outputs.last_hidden_state[:, 1, :]
    applicability.append(applicability_output)

    # Use the output to extract requirements
    requirements_output = outputs.last_hidden_state[:, 2, :]
    requirements.append(requirements_output)

    # ... (rest of the code remains the same)
    timelines, applicability, requirements = model(**inputs)
    processed_output = process_llm_output(timelines, applicability, requirements)
    return processed_output
