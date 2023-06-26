from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="opt-125m")   # default is the smallest model
    args = parser.parse_args()
    return args


args = get_args()
model_name = args.model_name

# always use fp16 on GPU
model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}", device_map="auto", torch_dtype=torch.float16, torchscript=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")

prompt = "Whatever you want is"
inputs = tokenizer(prompt)

# Trace the model
traced_model = torch.jit.trace(model, (torch.tensor([inputs.input_ids]), torch.tensor([inputs.attention_mask])))
torch.jit.save(traced_model, f"{model_name}.pt")
