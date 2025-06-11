import os
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import guidance
from guidance import models, gen, select


@guidance(stateless=True)
def few_shot(lm, lemma, usage1, usage2):
    return (
        lm
        + f"""
    You are an expert linguist analyzing lexical semantics problems.
    You are presented with two sentences that both contain a specific word.
    Your task is to analyze how this word is used in each sentence and determine if its usage in the second sentence represents the same sense with respect to its use in the first sentence.
    Follow these steps to complete the task:
        Step 1. Describe the meaning of the word in the first sentence.
        Step 2. Describe the meaning of the word in the second sentence.
        Step 3. Write a sentence that joins both sentences using zeugma and the same shared word while preserving the same sense.
            If the construction doesn't make a bad pun, the words have identical sense.
    Examples:
    Lemma: Plane
    Context 1: He loves planes and want to become a pilot.
    Context 2: The plane landed just now.
    Answer: identical
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    Answer: {select(["identical", "different"], "answer")}
    """
    )


@guidance(stateless=True)
def rhetorical(lm, lemma, usage1, usage2):
    return (
        lm
        + f"""
    You are an expert linguist analyzing lexical semantics problems.
    You are presented with two sentences that both contain a specific word.
    Your task is to analyze how this word is used in each sentence and determine if its usage in the second sentence represents the same sense with respect to its use in the first sentence.
    Follow these steps to complete the task:
        Step 1. Rewrite the first sentence in a simplified form.
        Step 2. Rewrite the second sentence in a simplified form.
        Step 3. Write a sentence that joins both sentences using zeugma and the same shared word.
            If the construction doesn't make a bad pun, the words have identical sense.
    Examples:
    Lemma: Plane
    Context 1: He loves planes and want to become a pilot.
    Context 2: The plane landed just now.
    1) He loves planes.
    2) The plane landed.
    3) He loves planes, like the one that landed.
    Conclusion: It doesn't make a bad pun so both sentences uses 'plane' with the same meaning.
    Answer: identical
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    1) {gen(stop="\n")}
    2) {gen(stop="\n")}
    3) {gen(stop="\n")}
    Conclusion: {gen(stop="\n")}
    Answer: {select(["identical", "different"], "answer")}
    """
    )


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        prog="prompt_generate.py",
        description="What the program does",
        epilog="""Generate the prompt for querying an LLM.

    Usage:
        prompt_generate.py [-n] <c1> <c2> <targets>

    Arguments:

        <corpus> = corpus
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <reasoning> = prompt the model to reason before answering

    """,
    )
    parser.add_argument("dataset")
    parser.add_argument("prompt")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(raw_args)
    model = models.LlamaCpp(
        args.model,
        # n_ctx=args.ctx,
        # n_gpu_layers=-1,
        # flash_attn=True,
        # echo=False,
    )

    with open(f"data/{args.dataset}.test.json") as fin:
        data = json.load(fin)

    path = "output"
    os.makedirs(path, exist_ok=True)

    pred = []
    true = []
    progress_bar = tqdm(data)
    for example in progress_bar:
        lm = model
        lm += rhetorical(example["LEMMA"], example["USAGE_x"], example["USAGE_y"])
        pred.append(lm["answer"])
        true.append(example["LABEL"])
        p, r, fscore, _ = precision_recall_fscore_support(pred, true)
        progress_bar.set_description(f"fscore: {fscore:.3f}")
        with open(path + "/{args.prompt}.jsonl", "w+") as fout:
            fout.write(str(lm))


if __name__ == "__main__":
    main()
