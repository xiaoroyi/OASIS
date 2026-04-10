import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import utils

ACCESS_TOKEN = utils.load_hf_token()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_folder,
        cache_dir=args.cache_dir,
        use_fast=True,
        token=ACCESS_TOKEN,
    )
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_folder,
        cache_dir=args.cache_dir,
        load_in_8bit=False,
        device_map="auto",
        torch_dtype=torch.float16,
        token=ACCESS_TOKEN,
    )
    if "Qwen2-7B" in args.model_folder:
        model = model.to(torch.bfloat16)

    if args.lora_folder:
        model = PeftModel.from_pretrained(model, args.lora_folder, torch_dtype=torch.float16)
        model = model.merge_and_unload()

    if args.lora_folder2:
        model = PeftModel.from_pretrained(model, args.lora_folder2, torch_dtype=torch.float16)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def build_input_data(args):
    instruction_list = []
    input_list = []

    if "BeaverTails" in args.instruction_path:
        dataset = load_dataset(args.beaverTails_dataset_path)
        input_data_list = []
        for example in dataset["30k_test"]:
            if len(input_data_list) >= 500:
                break
            if example["is_safe"]:
                continue
            input_data_list.append({"instruction": example["prompt"]})
            instruction_list.append(example["prompt"])
        return input_data_list, instruction_list, input_list

    with open(args.instruction_path, "r", encoding="utf-8") as f:
        input_data_list = json.load(f)
    for item in input_data_list:
        instruction_list.append(item["instruction"])
    return input_data_list, instruction_list, input_list


def query(model, tokenizer, instruction):
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1.0,
            temperature=1.0,
            do_sample=False,
            num_beams=1,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return output.split("### Response:")[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default="wxjiao/alpaca-7b")
    parser.add_argument("--lora_folder", default="")
    parser.add_argument("--lora_folder2", default="")
    parser.add_argument("--instruction_path", default="BeaverTails")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--cache_dir", default="../../cache")
    parser.add_argument("--beaverTails_dataset_path", default="PKU-Alignment/BeaverTails")
    args = parser.parse_args()

    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)

    input_data_list, instruction_list, input_list = build_input_data(args)
    model, tokenizer = load_model_and_tokenizer(args)

    predictions = []
    for instruction in tqdm(instruction_list):
        predictions.append(query(model, tokenizer, instruction))

    output_list = []
    for item, prediction in zip(input_data_list, predictions):
        item["output"] = prediction
        output_list.append(item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4)


if __name__ == "__main__":
    main()
