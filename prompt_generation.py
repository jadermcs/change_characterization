import os
import argparse
import json
import pandas as pd
from transformers import set_seed
from tqdm import tqdm
from guidance import models, gen, select

TEMP = 0.7


def prompt_gen(data, style=0):
    prompt = data["prompt"][style] + "\n"
    for ex in data["examples"]:
        prompt += "---\n"
        prompt += f"Word: {ex['word']}\n"
        prompt += "Sentences:\n"
        prompt += f"1) {ex['e1']}\n"
        prompt += f"2) {ex['e2']}\n"
        prompt += f"1. {ex['r1']}\n"
        prompt += f"2. {ex['r2']}\n"
        if style == 0:
            prompt += f"3. {ex['c']}\n"
        prompt += f"A: {ex['answer']}\n"
    return prompt


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
    parser.add_argument('--style', type=int)
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

    group = corpus[["WORD", "USAGE_1", "USAGE_2"]]
    # index, word, sentence a, sentence b
    path = f"output/{args.model}/{args.task}/{args.style}"
    os.makedirs(path, exist_ok=True)
    for idx, w, a, b in tqdm(list(group.itertuples())):
        text = prompt_gen(data[args.task], args.style)
        text += "---\n"
        text += f"Word: {w.replace('_', ' ').capitalize()}\n"
        text += "Sentences:\n"
        text += f"1) {a}\n"
        text += f"2) {b}\n"
        text += "<think>"
        lm = model + text
        lm += gen(temperature=TEMP, stop="</think>", max_tokens=4096)
        lm += "</think>"
        lm += "\nA: " + select(labels)
        with open(f"{path}/{args.seed}_{idx}.txt", "w") as fout:
            fout.write(str(lm))


if __name__ == '__main__':
    main()
