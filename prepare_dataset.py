import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

data_card = "mollahsabbir/nd400"
ds = load_dataset(data_card, split="validation")

evaluation_data_root = "./nd400_data"

print("Loading docci")
docci = load_dataset("google/docci")
docci_test = docci["test"]
docci_index = {ex["example_id"]: ex for ex in docci_test}

print("Loading nocaps")
nocaps = load_dataset("HuggingFaceM4/NoCaps", trust_remote_code=True)
nocaps_val = nocaps["validation"]
nocaps_index = {ex["image_file_name"]: ex for ex in nocaps_val}

def ensure_dirs(root, gen_index=0):
    captions_dir = os.path.join(root, "text-first")
    images_dir = os.path.join(root, "image-first")
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    captions_gen_dir = os.path.join(captions_dir, f"gen-{gen_index}")
    images_gen_dir = os.path.join(images_dir, f"gen-{gen_index}")
    os.makedirs(captions_gen_dir, exist_ok=True)
    os.makedirs(images_gen_dir, exist_ok=True)
    return captions_dir, images_dir, captions_gen_dir, images_gen_dir

def load_example(source, image_id):
    if source == "docci":
        ex = docci_index.get(image_id)
        if ex is None:
            return None
        return ex["image"], ex["description"], f"{image_id}.png"
    else:
        ex = nocaps_index.get(image_id)
        if ex is None:
            return None
        return ex["image"], ex["annotations_captions"][0], f"{image_id}.png"

def prepare_evaluation(dataset, root):
    captions_dir, images_dir, captions_gen_dir, images_gen_dir = ensure_dirs(root)
    rows = []
    for row in tqdm(dataset, desc="Processing ND400 examples"):
        result = load_example(row["source"], row["image_id"])
        if result is None:
            continue
        image, caption, filename = result

        image.save(os.path.join(images_gen_dir, filename))

        rows.append({"filename": filename, "caption": caption})
    pd.DataFrame(rows).to_csv(os.path.join(captions_dir, "gen-0.csv"), index=False)

if __name__ == "__main__":
    prepare_evaluation(ds, evaluation_data_root)
    print("Evaluation data prepared!")
