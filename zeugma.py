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
    Lemma: Cell
    Context 1: Anyone leaves a cell phone or handheld at home, many of them faculty members from nearby.
    Context 2: I just watch the dirty shadow the window bar makes across the wall of my cell.
    Answer: different
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    <think>
    {gen(stop="\n")}
    </think>
    Answer: {select(["identical", "different"], "answer")}
    """
    )


@guidance(stateless=True)
def rhetorical(lm, lemma, usage1, usage2):
    return (
        lm
        + f"""
    You are an expert linguist.
    You are presented with two sentences that both contain a shared word.
    Your task is to create and analyze zeugmas.
    Follow these steps to complete the task:
        Step 1. Rewrite the first sentence in a simplified form preserving the lemma.
        Step 2. Rewrite the second sentence in a simplified form preserving the lemma.
        Step 3. Write a sentence that joins both sentences using zeugma and the same shared word.
            If the construction doesn't make a bad pun, write same, otherwise, write different.
    Examples:
    Lemma: Plane
    Context 1: He loves planes and want to become a pilot.
    Context 2: The plane landed just now.
    1) He loves planes.
    2) The plane landed.
    3) He loves planes, like the one that landed.
    Conclusion: It doesn't make a bad pun.
    Answer: same
    ---
    Lemma: Cell
    Context 1: Anyone leaves a cell phone or handheld at home, many of them faculty members from nearby.
    Context 2: I just watch the dirty shadow the window bar makes across the wall of my cell.
    1) Anyone leaves a cell phone at home.
    2) The wall of my cell.
    3) The wall of my cell which I leave at home.
    Conclusion: It makes a bad pun.
    Answer: different
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    <think>
    {gen(stop='</think>')}
    </think>
    1) {gen("s1", stop="\n")}
    2) {gen("s2", stop="\n")}
    3) {gen("s3", stop="\n")}
    Conclusion: {gen("conclude", stop="\n")}
    Answer: {select(["same", "different"], "answer")}
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
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(raw_args)
    model = models.LlamaCpp(
        args.model,
        n_ctx=args.ctx,
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
        p, r, fscore, _ = precision_recall_fscore_support(
            true, pred, average="weighted"
        )
        progress_bar.set_description(f"fscore: {fscore:.3f}")
        output = {
            "s1": lm["s1"],
            "s2": lm["s2"],
            "s3": lm["s3"],
            "conclude": lm["conclude"],
            "answer": lm["answer"],
        }
        with open(path + f"/{args.prompt}.jsonl", "a") as fout:
            fout.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    main()
