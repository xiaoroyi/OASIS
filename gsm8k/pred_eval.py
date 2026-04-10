import argparse
import json
import os
import re
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
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"


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
        torch_dtype=torch.float16,
        device_map="auto",
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


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", sentence)
    if not matches:
        return float("inf")

    parts = sentence.split(ANSWER_PROMPT)
    if len(parts) > 1:
        answer_matches = re.findall(r"-?\d+\.?\d*", parts[1])
        candidate = answer_matches[0] if answer_matches else matches[-1]
    else:
        candidate = matches[-1]

    try:
        return float(candidate)
    except ValueError:
        return float("inf")


def query(model, tokenizer, item):
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{item['instruction']}\n\n"
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
    parser.add_argument("--output_path", default="../data/gsm8k/predictions.json")
    parser.add_argument("--cache_dir", default="../cache")
    parser.add_argument("--gsm8k_path", default="gsm8k")
    parser.add_argument("--gsm8k_config", default="main")
    args = parser.parse_args()

    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)

    dataset = load_dataset(args.gsm8k_path, args.gsm8k_config)
    input_data_list = []
    for example in dataset["test"]:
        if len(input_data_list) >= 500:
            break
        input_data_list.append(
            {
                "instruction": f"{example['question']}{QUESTION_PROMPT}",
                "output": example["answer"].replace("####", ANSWER_PROMPT),
            }
        )

    model, tokenizer = load_model_and_tokenizer(args)
    predictions = [query(model, tokenizer, item) for item in tqdm(input_data_list)]

    output_list = []
    correct = 0
    for item, prediction in zip(input_data_list, predictions):
        target = extract_answer_number(item["output"])
        predicted = extract_answer_number(prediction)
        is_correct = target == predicted
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
