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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import utils

ACCESS_TOKEN = utils.load_hf_token()


def load_sst2_dataset(dataset_name: str, dataset_config: str):
    if os.path.exists(dataset_name):
        return load_dataset(dataset_name)
    return load_dataset(dataset_name, dataset_config)


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


def query(model, tokenizer, item):
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Input:\n{item['input']}\n\n"
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
            max_new_tokens=200,
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
    parser.add_argument("--output_path", default="../data/sst2/predictions.json")
    parser.add_argument("--cache_dir", default="../cache")
    parser.add_argument("--sst2_path", default="glue")
    parser.add_argument("--sst2_config", default="sst2")
    args = parser.parse_args()

    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)

    dataset = load_sst2_dataset(args.sst2_path, args.sst2_config)
    input_data_list = []
    for example in dataset["validation"]:
        if len(input_data_list) >= 500:
            break
        input_data_list.append(
            {
                "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
                "input": example["sentence"],
                "label": example["label"],
            }
        )

    model, tokenizer = load_model_and_tokenizer(args)

    predictions = [query(model, tokenizer, item) for item in tqdm(input_data_list)]

    output_list = []
    correct = 0
    for item, prediction in zip(input_data_list, predictions):
        label = "positive" if item["label"] else "negative"
        prediction_lower = prediction.lower()
        is_correct = label in prediction_lower
        correct += int(is_correct)
        item["output"] = prediction
        item["correct"] = "true" if is_correct else "false"
        output_list.append(item)

    accuracy = 100 * correct / max(len(output_list), 1)
    print(f"{accuracy:.2f}")
    output_list.append(f"score={accuracy:.2f}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4)


if __name__ == "__main__":
    main()
