import os
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
from PIL import Image
from transformers import (
    CLIPTokenizer, CLIPModel, CLIPProcessor,
    AutoProcessor, AutoModel, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
import timm
from torchvision import transforms


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
        image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        return (image_feat @ text_feat.T).squeeze()


class CLIPWrapper(VisionLanguageModel):
    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_text_features(self, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.model.get_text_features(**inputs)

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs)


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
        self.model = timm.create_model("vit_base_patch16_224_dino", pretrained=True).to(device).eval()
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
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(tensor)
            if isinstance(feats, tuple):
                feats = feats[0]
            return feats


class BERTWrapper(VisionLanguageModel):
    def __init__(self, model_id="bert-base-uncased", device="cpu"):
        super().__init__(model_id, device)
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.model = BertModel.from_pretrained(model_id).to(device)

    def get_text_features(self, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.model(**inputs).pooler_output

    def get_image_features(self, image):
        raise NotImplementedError("BERT does not support image features.")


class SentenceTransformerWrapper(VisionLanguageModel):
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        super().__init__(model_id, device)
        self.model = SentenceTransformer(model_id).to(device)

    def get_text_features(self, captions):
        return self.model.encode(captions, convert_to_tensor=True).to(self.device)

    def get_image_features(self, image):
        raise NotImplementedError("SentenceTransformer is text-only.")


def open_annotation_file(csv_path):
    return pd.read_csv(csv_path).reset_index(drop=True)


def open_image(path):
    if os.path.exists(f"{path}.png"):
        return Image.open(f"{path}.png").convert("RGB")
    return Image.open(f"{path}.jpg").convert("RGB")


def compute_scores(annotation_ids, captions, model_wrapper, root_dir, baseline="text", generations=20):
    raw_scores = {g: {} for g in range(generations + 1)}

    if baseline == "text":
        baseline_features = [model_wrapper.get_text_features([c]) for c in captions]
    else:
        baseline_features = [model_wrapper.get_image_features(open_image(os.path.join(root_dir, "gen-0", str(a))))
                             for a in annotation_ids]

    for g in range(1, generations + 1):
        img_dir = os.path.join(root_dir, f"gen-{g}")
        csv_path = f"{img_dir}.csv"

        if os.path.exists(csv_path):  # text
            df = open_annotation_file(csv_path)
            new_feats = [model_wrapper.get_text_features([c]) for c in df["caption"].tolist()]
        else:  # image
            new_feats = [model_wrapper.get_image_features(open_image(os.path.join(img_dir, str(a))))
                         for a in annotation_ids]

        for i, ann in enumerate(annotation_ids):
            raw_scores[g][ann] = model_wrapper.similarity(new_feats[i], baseline_features[i]).item()

    return raw_scores


def mean_cumulative_drift(scores):
    drift = []
    for g in sorted(scores.keys())[1:]:
        drift.append(np.mean(list(scores[g].values())))
    return np.mean(drift)


def semantic_drift_rate(scores):
    gens = sorted(scores.keys())
    vals = [np.mean(list(scores[g].values())) for g in gens if g > 0]
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    return np.mean(diffs)
