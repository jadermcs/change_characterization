from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm


def initialize_vllm_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Initialize vLLM model with Llama3"""
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512,
        top_p=0.9,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    # Initialize the model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    return llm, sampling_params


def format_llama3_prompt(user_message: str) -> str:
    """Format prompt for Llama3 chat template"""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def generate_simple_prompt(
    target_word: str, pos: str, sentence1: str, sentence2: str
) -> str:
    user_message = f"""The word "{target_word}" appears as a {pos} in the following two sentences:

1. "{sentence1}"
2. "{sentence2}"

Task:
- Explain how the word is used in both cases.
- Answer: Do the two sentences use "{target_word}" in the same sense? [Yes/No]"""

    return format_llama3_prompt(user_message)


def generate_zeugma_prompt(
    target_word: str, pos: str, sentence1: str, sentence2: str
) -> str:
    user_message = f"""The word "{target_word}" appears as a {pos} in the following two sentences:

1. "{sentence1}"
2. "{sentence2}"

Try to construct a single sentence in which the word "{target_word}" governs both uses from the two sentences (as in a zeugma). If such a sentence is possible and the word retains a consistent meaning, the senses may be similar. If the sentence sounds awkward or the meaning of the word must shift between the two parts, the senses are likely different.

Task:
- Attempt to create such a sentence.
- Explain how the word is used in both cases.
- Answer: Do the two sentences use "{target_word}" in the same sense? [Yes/No]"""

    return format_llama3_prompt(user_message)


def query_vllm_batch(llm, sampling_params, prompts: list) -> list:
    """Query vLLM model with batch of prompts for efficiency"""
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def query_vllm_single(llm, sampling_params, prompt: str) -> str:
    """Query vLLM model with single prompt"""
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def parse_model_answer(response: str) -> str:
    """
    Parse model response to extract 'identical' or 'different' label.
    """
    response_lower = response.lower()
    if "yes" in response_lower:
        return "identical"
    elif "no" in response_lower:
        return "different"
    else:
        return "unknown"


def evaluate_wic_file(
    filepath: str,
    start_index: int,
    type: str = "zeugma",
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    batch_size: int = 8,
):
    """
    Evaluate WiC dataset using vLLM with Llama3

    Args:
        filepath: Path to the WiC JSON file
        start_index: Index to start evaluation from
        type: Type of prompt ("zeugma" or "simple")
        model_name: Llama3 model name/path
        batch_size: Batch size for inference (higher = faster but more memory)
    """
    # Initialize model
    print(f"Loading model: {model_name}")
    llm, sampling_params = initialize_vllm_model(model_name)

    # Load data
    df = pd.read_json(filepath)
    print(f"Loaded {len(df)} examples from {filepath}")

    predictions = []
    responses = []
    output = []

    # Prepare data subset
    subset_df = df.iloc[start_index:].reset_index(drop=True)

    try:
        # Process in batches for efficiency
        for batch_start in tqdm(
            range(0, len(subset_df), batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + batch_size, len(subset_df))
            batch_df = subset_df.iloc[batch_start:batch_end]

            # Generate prompts for batch
            batch_prompts = []
            for _, row in batch_df.iterrows():
                if type == "zeugma":
                    prompt = generate_zeugma_prompt(
                        row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
                    )
                else:
                    prompt = generate_simple_prompt(
                        row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
                    )
                batch_prompts.append(prompt)

            try:
                # Get batch responses
                batch_responses = query_vllm_batch(llm, sampling_params, batch_prompts)

                # Process responses
                for i, (_, row) in enumerate(batch_df.iterrows()):
                    response = batch_responses[i]
                    predicted_label = parse_model_answer(response)

                    responses.append(response)
                    predictions.append(predicted_label)
                    output.append(
                        {
                            "lemma": row["LEMMA"],
                            "pos": row["POS"],
                            "usage_x": row["USAGE_x"],
                            "usage_y": row["USAGE_y"],
                            "response": response,
                            "label": row["LABEL"],
                            "predicted_label": predicted_label,
                            "correct": row["LABEL"] == predicted_label,
                        }
                    )

            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                # Fallback to individual processing for this batch
                for _, row in batch_df.iterrows():
                    try:
                        if type == "zeugma":
                            prompt = generate_zeugma_prompt(
                                row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
                            )
                        else:
                            prompt = generate_simple_prompt(
                                row["LEMMA"], row["POS"], row["USAGE_x"], row["USAGE_y"]
                            )

                        response = query_vllm_single(llm, sampling_params, prompt)
                        predicted_label = parse_model_answer(response)

                    except Exception as inner_e:
                        print(f"Error on individual row: {inner_e}")
                        response = "unknown"
                        predicted_label = "unknown"

                    responses.append(response)
                    predictions.append(predicted_label)
                    output.append(
                        {
                            "lemma": row["LEMMA"],
                            "pos": row["POS"],
                            "usage_x": row["USAGE_x"],
                            "usage_y": row["USAGE_y"],
                            "response": response,
                            "label": row["LABEL"],
                            "predicted_label": predicted_label,
                            "correct": row["LABEL"] == predicted_label,
                        }
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print("Saving results...")

    # Convert to DataFrame and calculate metrics
    output_df = pd.DataFrame(output)

    if len(output_df) > 0:
        accuracy = output_df["correct"].sum() / len(output_df)
        print(
            f"\n✅ Accuracy: {accuracy:.2%} ({output_df['correct'].sum()}/{len(output_df)})"
        )

        # Show breakdown by label
        print("\nBreakdown by true label:")
        for label in output_df["label"].unique():
            subset = output_df[output_df["label"] == label]
            acc = subset["correct"].sum() / len(subset)
            print(f"  {label}: {acc:.2%} ({subset['correct'].sum()}/{len(subset)})")

        # Save results
        output_filename = f"wic_vllm_{type}_results.jsonl"
        output_df.to_json(output_filename, orient="records", lines=True)
        print(f"\nResults saved to: {output_filename}")
    else:
        print("No results to save.")

    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate WiC dataset using vLLM with Llama3"
    )
    parser.add_argument(
        "--file", default="data/wic.test.json", help="Path to WiC JSON file"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index for evaluation"
    )
    parser.add_argument(
        "--type", choices=["zeugma", "simple"], default="zeugma", help="Prompt type"
    )
    parser.add_argument(
        "--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name/path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )

    args = parser.parse_args()

    # Run evaluation
    result_df = evaluate_wic_file(
        filepath=args.file,
        start_index=args.start_index,
        type=args.type,
        model_name=args.model,
        batch_size=args.batch_size,
    )
