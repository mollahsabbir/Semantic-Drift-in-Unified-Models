import os
import torch
import pickle
import argparse
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPProcessor
from transformers import AutoProcessor, AutoModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
from IPython.display import display
import torch.nn.functional as F
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np 
from collections import defaultdict
import textwrap
from abc import ABC, abstractmethod
from transformers import (
    CLIPTokenizer, CLIPModel, CLIPProcessor,
    AutoProcessor, AutoModel,
    BertTokenizer, BertModel
)
from torchvision import transforms
from PIL import Image
import timm
from torchvision.transforms import ToPILImage
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from scipy.optimize import curve_fit

TEXT_FIRST = "text-first"
IMAGE_FIRST = "image-first"
MAX_GENERATIONS = 20

def open_annotation_file(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    df = df.sort_values(by='filename')
    df = df.reset_index(drop=True)
    return df

class VisionLanguageModel(ABC):
    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device

    @abstractmethod
    def get_text_features(self, captions):
        pass

    @abstractmethod
    def get_image_features(self, image):
        pass

    def similarity(self, image_feat, text_feat):
        # Default to cosine similarity
        image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        return (image_feat @ text_feat.T).squeeze()

class CLIPWrapper(VisionLanguageModel):
    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_text_features(self, captions):
        text_inputs = self.tokenizer(captions, padding="max_length", return_tensors="pt", truncation=True).to(self.device)
        return self.text_encoder.get_text_features(**text_inputs)

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self.text_encoder.get_image_features(**inputs)

class SigLIPWrapper(VisionLanguageModel):
    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)

    def get_text_features(self, captions):
        inputs = self.processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.model.get_text_features(**inputs)

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)

class DINOWrapper(VisionLanguageModel):
    def __init__(self, model_id="facebook/dinov2-base", device="cpu"):
        super().__init__(model_id, device)
        self.model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_text_features(self, captions):
        raise NotImplementedError("DINO does not support text features.")

    def get_image_features(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(img_tensor)  # shape: [1, 197, 768]
            if isinstance(feats, tuple):  # some models return (feats, _)
                feats = feats[0]
            return feats


class BERTWrapper(VisionLanguageModel):
    def __init__(self, model_id="bert-base-uncased", device="cpu"):
        super().__init__(model_id, device)
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.model = BertModel.from_pretrained(model_id).to(device)

    def get_text_features(self, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # [CLS] token representation

    def get_image_features(self, image):
        raise NotImplementedError("BERT does not support image features.")
    
class SentenceTransformerWrapper(VisionLanguageModel):
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        super().__init__(model_id, device)
        self.model = SentenceTransformer(model_id)
        self.model.to(device)

    def get_text_features(self, captions):
        # encode returns a tensor if convert_to_tensor=True
        embeddings = self.model.encode(captions, convert_to_tensor=True)
        return embeddings.to(self.device)

    def get_image_features(self, image):
        raise NotImplementedError("SentenceTransformer is text-only.")
    
class MPNetWrapper(VisionLanguageModel):
    def __init__(self, model_id="sentence-transformers/all-mpnet-base-v2", device="cpu"):
        super().__init__(model_id, device)
        self.model = SentenceTransformer(model_id)
        self.model.to(device)

    def get_text_features(self, captions):
        embeddings = self.model.encode(captions, convert_to_tensor=True)
        return embeddings.to(self.device)

    def get_image_features(self, image):
        raise NotImplementedError("MPNet is text-only.")

def get_scorer(scorer_name, device):
    siglip_model_id = "microsoft/siglip-base"
    clip_model_id = 'openai/clip-vit-base-patch32'
    if scorer_name == "clip":
        return CLIPWrapper(clip_model_id, device)
    elif scorer_name == "siglip":
        return SigLIPWrapper("microsoft/siglip-base", device)
    elif scorer_name == "dino":
        return DINOWrapper(device=device)
    elif scorer_name == "bert":
        return BERTWrapper(device=device)
    elif scorer_name == "mpnet":
        return MPNetWrapper(device=device)
    elif scorer_name == "sbert":
        return SentenceTransformerWrapper(device=device)
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}")

def open_image(image_id, img_dir):
    """
    Sometimes an image_id is stored as .png, sometimes as .jpg, sometimes without extension.
    This function handles all cases.
    """
    img_path = os.path.join(img_dir, str(image_id))
    if ".png" in img_path:
        img_path = img_path.replace(".png", "")
    
    if ".jpg" in img_path:
        img_path = img_path.replace(".jpg", "")
    
    if os.path.exists(f"{img_path}.png"):
        image = Image.open(f"{img_path}.png").convert("RGB")
    else:
        image = Image.open(f"{img_path}.jpg").convert("RGB")
        
    return image

def full_result_name(model_name, eval_type, from_modality, to_modality, scorer_name):
    filename = f"{model_name}_{eval_type}_{from_modality}_to_{to_modality}_with_{scorer_name}.pkl"
    return filename


def get_image_dir_at_gen(data_dir, eval_type, gen):
    return os.path.join(data_dir, eval_type, f"gen-{gen}")

def get_csv_dir_at_gen(data_dir, eval_type, gen):
    return os.path.join(data_dir, eval_type, f"gen-{gen}.csv")

def get_gen_range(eval_type, to_modality, max_generations=MAX_GENERATIONS):
    if eval_type == TEXT_FIRST:
        return range(2, max_generations + 1, 2) if to_modality == "text" else range(1, max_generations, 2)
    elif eval_type == IMAGE_FIRST:
        return range(2, max_generations + 1, 2) if to_modality=="image" else range(1, max_generations, 2)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

def image_first_scores(eval_type, data_dir, to_modality, scorer, max_generations=MAX_GENERATIONS):
    gen0_dir = get_image_dir_at_gen(data_dir, eval_type, gen=0)
    captions_path = get_csv_dir_at_gen(data_dir, eval_type, gen=1)
    df = open_annotation_file(captions_path)
    filenames = df["filename"].tolist()

    raw_scores = {gen:{fname:0 for fname in filenames} for gen in range(max_generations+1)}
    baseline_features = [scorer.get_image_features(open_image(f, gen0_dir).convert("RGB")) for f in filenames]
    print("Finished Baseline Features Extraction")

    gen_range = get_gen_range(eval_type, to_modality, max_generations)
    
    scores = {fname:{} for fname in filenames}

    for gen in gen_range:
        if to_modality=="image":
            gen_dir = get_image_dir_at_gen(data_dir, eval_type, gen)
            for i, fname in enumerate(tqdm(filenames, desc=f"Gen-{gen}")):
                img = open_image(fname, gen_dir).convert("RGB")
                feat = scorer.get_image_features(img)
                sim = scorer.similarity(feat, baseline_features[i])
                scores[fname][gen] = sim.item()
                raw_scores[gen][fname] = sim.item()
        else:
            csv_path = get_csv_dir_at_gen(data_dir, eval_type, gen)
            df_gen = open_annotation_file(csv_path)
            captions_gen = df_gen["caption"].tolist()
            for i, fname in enumerate(tqdm(filenames, desc=f"Gen-{gen}")):
                feat = scorer.get_text_features([captions_gen[i]])
                sim = scorer.similarity(feat, baseline_features[i])
                scores[fname][gen] = sim.item()
                raw_scores[gen][fname] = sim.item()

    gen_mean_scores = []
    for gen in gen_range:
        gen_scores = [scores[fname][gen] for fname in filenames if gen in scores[fname]]
        if gen_scores:
            mean_score = np.mean(gen_scores)
            gen_mean_scores.append(mean_score)
            print(f"Generation {gen} mean similarity score: {mean_score:.4f}")
        else:
            print(f"Generation {gen} has no scores.")

    return scores, raw_scores, gen_mean_scores


def text_first_scores(eval_type,
                          data_dir, 
                          to_modality,
                          scorer,
                          max_generations=MAX_GENERATIONS):

    captions_path = os.path.join(data_dir, eval_type, "gen-0.csv")
    captions_df = open_annotation_file(captions_path)
    captions = captions_df["caption"].tolist()
    filenames = captions_df["filename"].tolist()

    raw_scores = {gen:{filename:0 for filename in filenames} for gen in range(max_generations+1)}

    # Compute features for original captions (gen-0)
    baseline_features = [scorer.get_text_features([caption]) for caption in captions]
    print("Finished Baseline Features Extraction")

    # Determine generation indices to score
    gen_range = get_gen_range(eval_type, to_modality, max_generations)

    scores = {fname: {} for fname in filenames}

    for gen in gen_range:
        if to_modality == "image":
            image_dir = get_image_dir_at_gen(data_dir, eval_type, gen)
            print(f"Processing {image_dir}")
            for i, filename in enumerate(tqdm(filenames, desc=f"Gen-{gen}")):
                image = open_image(filename, image_dir).convert("RGB")
                img_feat = scorer.get_image_features(image)
                sim = scorer.similarity(img_feat[0], baseline_features[i])
                scores[filename][gen] = sim.item()
                raw_scores[gen][filename] = sim.item()

        elif to_modality == "text":
            csv_path = os.path.join(data_dir, eval_type, f"gen-{gen}.csv")
            captions_df_gen = open_annotation_file(csv_path)
            captions_gen = captions_df_gen["caption"].tolist()
            for i, filename in enumerate(tqdm(filenames, desc=f"Gen-{gen}")):
                text_feat = scorer.get_text_features([captions_gen[i]])
                sim = scorer.similarity(text_feat, baseline_features[i])
                scores[filename][gen] = sim.item()
                raw_scores[gen][filename] = sim.item()

    gen_mean_scores = []
    # Compute and print mean per generation
    for gen in gen_range:
        gen_scores = [scores[fname][gen] for fname in filenames if gen in scores[fname]]
        if gen_scores:
            mean_score = np.mean(gen_scores)
            gen_mean_scores.append(mean_score)
            print(f"Generation {gen} mean similarity score: {mean_score:.4f}")
        else:
            print(f"Generation {gen} has no scores.")

    return scores, raw_scores, gen_mean_scores


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def plot_and_save(x_data, y_data, params, title, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x_data, y_data, label="Mean scores", color="blue")
    if params is not None and not any(np.isnan(params)):
        a, b, c = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = exp_decay(x_fit, a, b, c)
        plt.plot(x_fit, y_fit, label=f"Fit: a={a:.2f}, b={b:.2f}, c={c:.2f}", color="red", linestyle=":")
    plt.xlabel("Generation")
    plt.ylabel("Similarity")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def main():
    evaluations = [
        {"eval_type": TEXT_FIRST, "from": "text", "to": "text", "scorer": "mpnet"},
        {"eval_type": TEXT_FIRST, "from": "text", "to": "image", "scorer": "clip"},
        {"eval_type": IMAGE_FIRST, "from": "image", "to": "image", "scorer": "dino"},
        {"eval_type": IMAGE_FIRST, "from": "image", "to": "text", "scorer": "clip"},
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to evaluation data root")
    parser.add_argument("--out", required=True, help="Output directory for plots and pickle files")
    parser.add_argument("--model-name", required=True, help="Model name is used to name output files")
    args = parser.parse_args()

    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_raw_scores = []
    all_params = []
    all_gen_means = []

    for evaluation in evaluations:
        eval_type = evaluation["eval_type"]
        from_modality = evaluation["from"]
        to_modality = evaluation["to"]
        scorer_name = evaluation["scorer"]

        scorer = get_scorer(scorer_name, device)
        result_name = full_result_name(model_name, eval_type, from_modality, to_modality, scorer_name)
        result_path = os.path.join(args.out, result_name)

        if os.path.exists(result_path):
            print(f"Skipping evaluation (already exists): {result_path}")
            with open(result_path, "rb") as f:
                sample_scores, raw_scores, gen_mean_scores = pickle.load(f)
        else:
            print(f"Running evaluation: {eval_type}, from {from_modality} to {to_modality}, with scorer {scorer_name}")
            if eval_type == TEXT_FIRST:
                sample_scores, raw_scores, gen_mean_scores = text_first_scores(eval_type, args.data, to_modality, scorer, max_generations=MAX_GENERATIONS)
            elif eval_type == IMAGE_FIRST:
                sample_scores, raw_scores, gen_mean_scores = image_first_scores(eval_type, args.data, to_modality, scorer, max_generations=MAX_GENERATIONS)
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")

            os.makedirs(args.out, exist_ok=True)
            with open(result_path, "wb") as f:
                pickle.dump((sample_scores, raw_scores, gen_mean_scores), f)

        print(f"MCD ({from_modality}-to-{to_modality}): {np.mean(gen_mean_scores):.4f}")
        all_gen_means.append(gen_mean_scores)
        all_raw_scores.append(raw_scores)

        x_data = list(get_gen_range(eval_type, to_modality, MAX_GENERATIONS))
        params, _ = curve_fit(exp_decay, x_data, gen_mean_scores, p0=(1, 0.1, 0), maxfev=5000)
        a, b, c = params
        all_params.append(params)
        print(f"{evaluation['from']}-to-{evaluation['to']} fitted params: a={a:.4f}, b={b:.4f}, c={c:.4f}")

        save_path = os.path.join(args.out, f"{result_name}.png")
        title = f"{from_modality}-to-{to_modality} ({scorer_name})"
        plot_and_save(x_data, gen_mean_scores, params, title, save_path)

    print(f"MCD: {np.mean([np.mean(g) for g in all_gen_means]):.4f}")

    all_params = np.array(all_params)
    avg_a, avg_b, avg_c = np.mean(all_params, axis=0)
    print(f"SDR: a={avg_a:.4f}, b={avg_b:.4f}, c={avg_c:.4f}")

    # Final averaged curve
    min_len = min(len(g) for g in all_gen_means)
    avg_curve = np.mean([g[:min_len] for g in all_gen_means], axis=0)
    x_data = list(range(1, min_len + 1))
    final_params, _ = curve_fit(exp_decay, x_data, avg_curve, p0=(1, 0.1, 0), maxfev=5000)
    plot_and_save(x_data, avg_curve, final_params, "Final Averaged SDR", os.path.join(args.out, "final.png"))



if __name__ == "__main__":
    main()