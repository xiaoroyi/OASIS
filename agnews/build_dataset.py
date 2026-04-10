import json
from pathlib import Path

from datasets import load_dataset


LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def main():
    dataset = load_dataset("ag_news")
    output_path = Path(__file__).resolve().parents[1] / "data" / "ag_news.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = []
    for data in dataset["train"]:
        output_data.append(
            {
                "instruction": (
                    "Categorize the news article given in the input into one of the 4 categories:\n\n"
                    "World\nSports\nBusiness\nSci/Tech\n"
                ),
                "input": data["text"],
                "output": LABEL_MAP[data["label"]],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
