import json
from pathlib import Path

from datasets import load_dataset


def main():
    dataset = load_dataset("glue", "sst2")
    output_path = Path(__file__).resolve().parents[1] / "data" / "sst2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = []
    for data in dataset["train"]:
        output_data.append(
            {
                "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
                "input": data["sentence"],
                "output": "positive" if data["label"] == 1 else "negative",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
