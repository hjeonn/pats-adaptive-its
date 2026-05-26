import torch
import logging
import openai
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import config

if config.OPENAI_API_KEY:
    openai.api_key = config.OPENAI_API_KEY
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)

def load_model(model_type):
    if model_type.startswith("gpt"):
        return {"model": model_type, "device": "api", "type": "api"}
    elif model_type.startswith("gemini"):
        return {"model": "gemini-2.5-flash", "device": "api", "type": "api"}
    elif model_type in ["t5-large", "llama-3.2-1b", "mistral-7b"]:
        bnb_config = None
        lora_config = None

        if model_type == "t5-large":
            model_name = "google/t5-v1_1-large"
            model_class = AutoModelForSeq2SeqLM
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM")
        elif model_type == "llama-3.2-1b":
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            model_class = AutoModelForCausalLM
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        elif model_type == "mistral-7b":
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            model_class = AutoModelForCausalLM
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.HF_TOKEN)
        
        model_kwargs = {"device_map": "auto", "token": config.HF_TOKEN}
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = model_class.from_pretrained(model_name, **model_kwargs)

        if lora_config:
            model = get_peft_model(model, lora_config)

        if model_type != "t5-large" and not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        return {"tokenizer": tokenizer, "model": model, "device": str(model.device), "type": "local"}
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_response(prompt, model_dict, model_type, temperature=0.7):
    if model_dict["type"] == "api":
        if model_type.startswith("gpt"):
            try:
                response = openai.ChatCompletion.create(
                    model=model_dict["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                logging.error(f"OpenAI API call error: {e}")
                return ""
        elif model_type.startswith("gemini"):
            try:
                client = genai.GenerativeModel(model_dict["model"])
                response = client.generate_content(
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config={"temperature": temperature}
                )
                return response.text
            except Exception as e:
                logging.error(f"Gemini API call error: {e}")
                return ""

    elif model_dict["type"] == "local":
        tokenizer = model_dict["tokenizer"]
        model = model_dict["model"]
        device = model_dict["device"]

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                if model_type == "t5-large":
                    output = model.generate(**inputs, max_length=512)
                    return tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    output = model.generate(
                        **inputs, 
                        max_length=512, 
                        do_sample=True, 
                        top_p=0.8, 
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    input_length = inputs.input_ids.shape[-1]
                    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Local model generation error: {e}")
            return ""