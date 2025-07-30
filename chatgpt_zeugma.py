from openai import OpenAI
import os
import argparse
import pandas as pd
from tqdm import tqdm

# Set your API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_hayashi(target_word: str, pos: str, sentence1: str, sentence2: str) -> str:
    return f"""Identify if the target word "{target_word}" in the following sentences correspond to the identical meanings or not.
    Answer Yes or No, and provide one brief sentence to describe the rationale behind the decision.
    1: {sentence1}
    2: {sentence2}
    Answer:"""

def generate_simple_prompt(
    target_word: str, pos: str, sentence1: str, sentence2: str
) -> str:
    return f"""The word "{target_word}" appears in the following two sentences:

1. "{sentence1}"
2. "{sentence2}"

Task:
- Explain how the word is used in both cases.
- Answer: Do the two sentences use "{target_word}" in the same sense? [Yes/No]
"""


def generate_zeugma_prompt(
    target_word: str, pos: str, sentence1: str, sentence2: str
) -> str:
    return f"""The word "{target_word}" appears in the following two sentences:

1. "{sentence1}"
2. "{sentence2}"

Try to construct a single sentence in which the word "{target_word}" governs both uses from the two sentences (as in a zeugma). If such a sentence is possible and the word retains a consistent meaning, the senses may be similar. If the sentence sounds awkward or the meaning of the word must shift between the two parts, the senses are likely different.

Task:
- Attempt to create such a sentence.
- Explain how the word is used in both cases.
- Answer: Do the two sentences use "{target_word}" in the same sense? [Yes/No]
"""


def query_openai(prompt: str, model="gpt-4o-mini", temperature=0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def parse_model_answer(response: str) -> str:
    """
    Parse model response to extract 'same' or 'different' label.
    """
    response_lower = response.lower()
    if "yes" in response_lower:
        return "identical"
    elif "no" in response_lower:
        return "different"
    else:
        return "unknown"


def evaluate_wic_file(filepath: str, start_index: int, type: str = "zeugma"):
    df = pd.read_json(filepath)
    predictions = []
    responses = []
    output = []

    for i, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df)):
        if type == "zeugma":
            prompt = generate_zeugma_prompt(
                row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
            )
        elif type == "hayashi":
            prompt = generate_hayashi(
                row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
            )
        else:
            prompt = generate_simple_prompt(
                row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
            )
        try:
            response = query_openai(prompt)
            predicted_label = parse_model_answer(response)
        except Exception as e:
            print(f"Error on row {i}: {e}")
            response = "unknown"
            predicted_label = "unknown"

        responses.append(response)
        predictions.append(predicted_label)

        output = pd.DataFrame(
            {
                "lemma": row["LEMMA"],
                "pos": row["POS"],
                "usage_x": row["USAGE_x"],
                "usage_y": row["USAGE_y"],
                "response": response,
                "label": row["LABEL"],
                "predicted_label": predicted_label,
                "correct": row["LABEL"] == predicted_label,
            },
            index=[i],
        ).reset_index()
        filename = f"wic_{type if type else 'simple'}_results.jsonl"
        output.to_json(
            filename,
            orient="records",
            lines=True,
            mode="a",
        )
    return filename


if __name__ == "__main__":
    # Replace with your file path
    args = argparse.ArgumentParser()
    args.add_argument("--type", type=str, help="Type of prompt", default="zeugma")
    args.add_argument(
        "--file_path", type=str, help="File path", default="data/wic.test.json"
    )
    args.add_argument("--start_index", type=int, help="Start index", default=0)
    args = args.parse_args()
    filename = evaluate_wic_file(
        args.file_path, start_index=args.start_index, type=args.type
    )
    data = pd.read_json(filename, lines=True)
    print("Accuracy:", (data["correct"].astype(bool)).sum() / data.shape[0])
