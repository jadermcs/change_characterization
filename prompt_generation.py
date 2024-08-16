import os
import argparse
import json
import pandas as pd
# import torch
from transformers import set_seed
from tqdm import tqdm
from guidance import models, gen, select

TEMP = 0.7


def compare(sample, example, bleu=None):
    maxbleu = .0
    best = None
    for i in sample:
        curbleu = bleu.sentence_score(i, [example]).score
        if curbleu > maxbleu:
            maxbleu = curbleu
            best = i
    return best


def gen_task(task):
    if task == "dimension":
        return ["identical", "different"]
    if task == "relation":
        return ["metonymy", "metaphor", "unrelated"]
    if task == "orientation":
        return ["positive", "negative", "neutral"]


def prompt_gen(data, reason, style=0):
    prompt = data["prompt"][style] + "\n"
    for ex in data["examples"]:
        prompt += "---\n"
        prompt += f"Word: {ex['word']}\n"
        prompt += "Sentences:\n"
        prompt += f"1) {ex['e1']}\n"
        prompt += f"2) {ex['e2']}\n"
        if reason and style == 0:
            prompt += "Analysis:\n"
            prompt += f"1. {ex['r1']}\n"
            prompt += f"2. {ex['r2']}\n"
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
    parser.add_argument('--reason', action='store_true')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args(raw_args)
    # model = models.Transformers(
    #         model_name_or_path,
    #         device="cuda",
    #         torch_dtype=torch.float16,
    #         echo=False)
    set_seed(args.seed)
    model = models.LlamaCpp(
            "models/"+args.model,
            n_gpu_layers=-1,
            n_ctx=args.ctx,
            flash_attn=True,
            echo=False)

    corpus = pd.read_csv(args.corpus, quoting=3, sep='\t')
    with open(args.instruction) as fin:
        data = json.load(fin)

    group = corpus[["word", "context1", "context2"]]
    if args.sample:
        group = group.sample(frac=.1, random_state=42)
    # index, word, sentence a, sentence b
    path = f"output/{args.style}" + ("" if args.reason else "_noreason")
    os.makedirs(path, exist_ok=True)
    for idx, w, a, b in tqdm(list(group.itertuples())):
        text = prompt_gen(data[args.task], args.reason, args.style)
        text += "---\n"
        text += f"Word: {w.replace('_', ' ').capitalize()}\n"
        text += "Sentences:\n"
        text += f"1) {a}\n"
        text += f"2) {b}\n"
        lm = model + text
        if args.style == 0:
            lm += "Analysis:\n"
            lm += "1. " + gen(suffix='\n', temperature=TEMP)
            lm += "2. " + gen(suffix='\n', temperature=TEMP)
            lm += "3. " + gen(suffix='\n', temperature=TEMP)
        if args.style == 1 and args.reason:
            lm += "Let's think step-by-step. " + gen(temperature=TEMP)
        else:
            lm += "\nA: " + select(gen_task(args.task))
        with open(f"{path}/{args.task}_{args.seed}_{args.model}_{idx}.txt", "w") as fout:
            fout.write(str(lm))


if __name__ == '__main__':
    main()
