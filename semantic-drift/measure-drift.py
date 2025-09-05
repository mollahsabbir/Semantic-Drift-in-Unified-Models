import argparse
import os
import torch
import pickle
from mcd_sdr import (
    CLIPWrapper, SigLIPWrapper, DINOWrapper, BERTWrapper,
    SentenceTransformerWrapper, open_annotation_file,
    compute_scores, mean_cumulative_drift, semantic_drift_rate
)
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["clip", "siglip", "bert", "sentence-transformers", "dino"]
EXPERIMENT_TYPES = ["captions-first", "images-first"]
GENERATIONS = 20


def get_model(name, device):
    if name == "clip":
        return CLIPWrapper("openai/clip-vit-base-patch32", device)
    if name == "siglip":
        return SigLIPWrapper("google/siglip2-base-patch32-256", device)
    if name == "bert":
        return BERTWrapper(device=device)
    if name == "sentence-transformers":
        return SentenceTransformerWrapper(device=device)
    if name == "dino":
        return DINOWrapper(device=device)
    raise ValueError(f"Unknown model {name}")


def save_boxplot(scores, out_file, generations=GENERATIONS):
    data = [[scores[g][fid] for fid in scores[g]] for g in range(1, generations+1)]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=[f"gen-{g}" for g in range(1, generations+1)], showmeans=True)
    plt.title("Similarity Scores per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def run_all(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    for exp_type in EXPERIMENT_TYPES:
        print(f"\n=== Experiment: {exp_type} ===")
        data_root = os.path.join(args.data, exp_type)

        # Load gen-0 CSV
        csv0 = os.path.join(data_root, "gen-0.csv")
        df = open_annotation_file(csv0)
        annotation_ids = df["filename"].tolist()
        captions = df["caption"].tolist()

        for model_name in MODELS:
            print(f"Processing model: {model_name}")
            model = get_model(model_name, device)

            # Compute similarity scores
            scores = compute_scores(annotation_ids, captions, model, data_root, baseline="text",
                                    generations=GENERATIONS)

            # Metrics
            mcd = mean_cumulative_drift(scores)
            sdr = semantic_drift_rate(scores)
            print(f"Model: {model_name}, MCD: {mcd:.4f}, SDR: {sdr:.4f}")

            # Save pickle
            pkl_file = os.path.join(args.out, f"{model_name}_{exp_type}_scores.pkl")
            with open(pkl_file, "wb") as f:
                pickle.dump(scores, f)

            # Save boxplot
            plot_file = os.path.join(args.out, f"{model_name}_{exp_type}_boxplot.png")
            save_boxplot(scores, plot_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", required=True, help="Path to evaluation data root")
    parser.add_argument("-out", required=True, help="Output folder for plots and pickle files")
    args = parser.parse_args()

    run_all(args)
