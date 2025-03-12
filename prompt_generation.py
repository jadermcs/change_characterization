import os
import argparse
import json
import pandas as pd
from transformers import set_seed
from tqdm import tqdm
import guidance
from guidance import models, gen, select


@guidance
def prompt_gen(lm, data, rhetorics=False):
    lm += data["system"]
    lm += data["prompt"]
    if rhetorics:
        lm += "\n" + data["rhetorics"]
    lm += "\nExamples:\n"
    for ex in data["examples"]:
        lm += "---\n"
        lm += f"Word: {ex['lemma']}\n"
        lm += "Sentences:\n"
        lm += f"1) {ex['e1']}\n"
        lm += f"2) {ex['e2']}\n"
        lm += f"1. {ex['r1']}\n"
        lm += f"2. {ex['r2']}\n"
        if rhetorics:
            lm += f"3. {ex['c']}\n"
        lm += f"Answer: {ex['answer']}\n"
    return lm


def main(raw_args=None):
    parser = argparse.ArgumentParser(
                    prog='prompt_generate.py',
                    description='What the program does',
                    epilog="""Generate the prompt for querying an LLM.

    Usage:
        prompt_generate.py [-n] <c1> <c2> <targets>

    Arguments:

        <corpus> = corpus
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <reasoning> = prompt the model to reason before answering

    """)
    parser.add_argument('corpus')
    parser.add_argument('instruction')
    parser.add_argument('task')
    parser.add_argument('--rhetorics', action='store_true')
    parser.add_argument('--ctx', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args(raw_args)
    set_seed(args.seed)
    model = models.LlamaCpp(
            "models/"+args.model,
            n_gpu_layers=-1,
            n_ctx=args.ctx,
            flash_attn=True,
            echo=False)

    corpus = pd.read_csv(args.corpus, quoting=3, sep='\t')
    labels = corpus.LABEL.unique().tolist()
    with open(args.instruction) as fin:
        data = json.load(fin)

    group = corpus[["LEMMA", "USAGE_1", "USAGE_2"]]
    style = 'rhetorics' if args.rhetorics else 'cot'
    path = f"output/{args.model}/{args.task}/{style}"
    os.makedirs(path, exist_ok=True)
    for idx, w, a, b in tqdm(list(group.itertuples())):
        lm = model + prompt_gen(data[args.task], args.rhetorics)
        lm += "---\n"
        lm += "Task:\n"
        lm += f"Word: {w.replace('_', ' ').capitalize()}\n"
        lm += "Sentences:\n"
        lm += f"1) {a}\n"
        lm += f"2) {b}\n"
        lm += "Let's think step by step. "
        lm += gen(max_tokens=1024)
        lm += "\nBased on my reasoning, here is my final answer:\n"
        lm += "Answer: " + select(labels)
        with open(f"{path}/{args.seed}_{idx}.txt", "w") as fout:
            fout.write(str(lm))


if __name__ == '__main__':
    main()
