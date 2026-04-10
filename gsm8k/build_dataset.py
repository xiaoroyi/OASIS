import json
from pathlib import Path

from datasets import load_dataset

ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"


def main():
    dataset = load_dataset("gsm8k", "main")
    output_path = Path(__file__).resolve().parents[1] / "data" / "gsm8k.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = []
    for data in dataset["train"]:
        output_data.append(
            {
                "instruction": f"{data['question']}{QUESTION_PROMPT}",
                "output": data["answer"].replace("####", ANSWER_PROMPT),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
