import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt


def main(raw_args=None):
    parser = argparse.ArgumentParser(
                    prog='compute_score.py',
                    description='What the program does',
                    epilog="""Generate the prompt for querying an LLM.

    Usage:
        .py [-n] <c1> <c2> <targets>

    Arguments:

    TODO

    """)
    parser.add_argument('data')
    parser.add_argument('seed')
    parser.add_argument('style')
    parser.add_argument('path')
    parser.add_argument('task')
    parser.add_argument('model')
    args = parser.parse_args(raw_args)

    df1 = pd.read_csv(args.data, sep="\t", quoting=3)

    real = []
    pred = []
    files = []

    path = os.path.join(args.path, args.model, args.task, args.style)

    for file in os.listdir(path):
        with open(path+file) as fin:
            data = fin.read()
            label = data.split("\n")[-1].removeprefix("A: ")
            _id = int(file.split("_")[0])
            files.append(file)
            real.append(df1.iloc[_id]["label"])
            pred.append(label)

    exp_name = f"results/{args.model}/{args.task}/{args.style}/{args.seed}"
    with open(f"{exp_name}.txt", "w") as fout:
        fout.write(classification_report(real, pred))
    out = pd.DataFrame({"file": files, "real": real, "predict": pred})
    out.to_csv(f"{exp_name}.csv")

    print(accuracy_score(real, pred))
    cm = confusion_matrix(real, pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.viridis, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center',
                    size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(f"{exp_name}.png")


if __name__ == "__main__":
    main()
