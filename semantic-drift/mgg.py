import json
import os
from tqdm import tqdm
import argparse
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from transformers import Owlv2ForObjectDetection, Owlv2Processor
import torch
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script requires a CUDA-enabled GPU."
POSITION_THRESHOLD = 0.1

def show_score_summary(all_accuracies):
    aggregated = defaultdict(lambda: {"correct": 0, "total": 0})

    for gen_acc in all_accuracies:
        for task, stats in gen_acc.items():
            aggregated[task]["correct"] += stats.get("correct", 0)
            aggregated[task]["total"] += stats.get("total", 0)

    print("\n=== Score Summary Across Generations ===")
    for task, stats in aggregated.items():
        correct = stats["correct"]
        total = stats["total"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        print(f"{task:15s}: {correct}/{total} correct, Accuracy: {accuracy:.2f}%")


def load_owlv2():
    model_card = "google/owlv2-large-patch14-ensemble"
    processor = Owlv2Processor.from_pretrained(model_card)
    model = Owlv2ForObjectDetection.from_pretrained(model_card).to(DEVICE)

    return processor, model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OWLv2 detections with metadata.")

    parser.add_argument("--imagedir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--metafile", type=str, required=True, help="Path to metadata .jsonl file")
    parser.add_argument("--resultname", type=str, required=True, help="Name to tag in results")
    parser.add_argument("--resultdir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold for filtering detections")

    return parser.parse_args()

def relative_position(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    cx1, cy1 = (xa1 + xa2) / 2, (ya1 + ya2) / 2
    cx2, cy2 = (xb1 + xb2) / 2, (yb1 + yb2) / 2
    w1, h1 = xa2 - xa1, ya2 - ya1
    w2, h2 = xb2 - xb1, yb2 - yb1
    dx, dy = cx1 - cx2, cy1 - cy2
    rx = max(abs(dx) - POSITION_THRESHOLD * (w1 + w2), 0) * np.sign(dx)
    ry = max(abs(dy) - POSITION_THRESHOLD * (h1 + h2), 0) * np.sign(dy)

    relations = set()
    if rx < -0.5: relations.add("left of")
    if rx > 0.5: relations.add("right of")
    if ry < -0.5: relations.add("above")
    if ry > 0.5: relations.add("below")
    return relations

@torch.no_grad()
def evaluate_image(image_path, metadata, processor, model, threshold=0.1):
    image = Image.open(image_path).convert("RGB")

    # Build prompts including color if specified
    classes = []
    for inc in metadata.get("include", []):
        cls = inc["class"]
        if "color" in inc:
            cls = f"{inc['color']} {cls}"
        classes.append(cls)

    # Prepare prompts for OWLv2
    texts = [[f"a photo of a {cls}"] for cls in classes]

    # Run OWLv2 model
    inputs = processor(text=texts, images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]]).to(model.device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    # Collect detections
    detections = defaultdict(list)
    all_boxes = defaultdict(list)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_idx = label.item()
        if label_idx >= len(classes):
            continue
        cls = metadata["include"][label_idx]["class"]  # Use original class name for indexing
        box = box.cpu().tolist()
        detections[cls].append({"score": float(score), "box": box})
        all_boxes[cls].append(box)

    # Evaluation logic
    is_correct = True
    reasons = []
    matched_groups = []

    for idx, req in enumerate(metadata.get("include", [])):
        cls = req["class"]
        required_count = req.get("count", 1)
        boxes = all_boxes.get(cls, [])
        used_boxes = []
        for box in boxes:
            # Find the corresponding detection with this box
            for det in detections[cls]:
                if det["box"] == box:
                    if det["score"] < threshold:
                        continue  # Skip boxes below threshold
                    used_boxes.append(box)
                    break

        if len(used_boxes) < required_count:
            is_correct = False
            reasons.append(f"expected {cls} ≥ {required_count} above score {threshold}, found {len(used_boxes)}")
            matched_groups.append(None)
            continue

        # Position check
        if "position" in req:
            relation, target_idx = req["position"]
            ref_group = matched_groups[target_idx]
            if not ref_group:
                is_correct = False
                reasons.append(f"no target for {cls} to be {relation}")
            else:
                for box_a in used_boxes:
                    if not any(relation in relative_position(box_a, box_b) for box_b in ref_group):
                        is_correct = False
                        reasons.append(f"{cls} not {relation} target")

        matched_groups.append(used_boxes)

    # Exclude logic
    for req in metadata.get("exclude", []):
        cls = req["class"]
        max_count = req.get("count", 999)
        actual = len(all_boxes.get(cls, []))
        if actual >= max_count:
            is_correct = False
            reasons.append(f"expected {cls} < {max_count}, found {actual}")

    return {
        "filename": os.path.basename(image_path),
        "correct": is_correct,
        "reason": "; ".join(reasons),
        "metadata": metadata,
        "detections": detections
    }

def compute_accuracies_from_results(results):
    from collections import defaultdict

    category_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        tag = result["metadata"].get("tag", "unknown")
        category_counts[tag]["total"] += 1
        if result["correct"]:
            category_counts[tag]["correct"] += 1

    # Calculate per-category and overall accuracy
    accuracy_report = {}
    total_correct = 0
    total_total = 0

    for tag, stats in category_counts.items():
        correct = stats["correct"]
        total = stats["total"]
        accuracy = correct / total if total > 0 else 0.0
        accuracy_report[tag] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy * 100, 2)
        }
        total_correct += correct
        total_total += total

    accuracy_report["overall"] = {
        "correct": total_correct,
        "total": total_total,
        "accuracy": round((total_correct / total_total) * 100, 2) if total_total > 0 else 0.0
    }

    return accuracy_report

 

def main():
    all_gen_results = []
    args = parse_args()
    all_gen_results = []

    resultdir = args.resultdir


    processor, model = load_owlv2()
    os.makedirs(resultdir, exist_ok=True)

    # Load metadata
    with open(args.metafile, "r") as f:
        metadata_list = [json.loads(line.strip()) for line in f if line.strip()]

    all_accuracies = []


    for gen in range(1,20,2):
        # print(f"Starting generation {gen}")
        # Define result filenames
        result_file = os.path.join(resultdir, f"{args.resultname}-gen{gen}.jsonl")
        accuracy_file = os.path.join(resultdir, f"{args.resultname}-accuracy.json")

        results = []

        # Load results from file if they exist
        if os.path.exists(result_file):
            print(f"Loading cached results from {result_file}")
            with open(result_file, "r") as f:
                results = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            gen_image_path = args.imagedir + f"gen-{gen}"

            # Evaluate each image
            for idx, metadata in enumerate(tqdm(metadata_list, desc=f"Evaluating Gen {gen}")):
                if idx == 4:
                    break
                image_path = os.path.join(gen_image_path, f"{idx}.png")
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}, skipping.")
                    continue
                result = evaluate_image(image_path, metadata, processor, model, args.threshold)
                results.append(result)

            # Save results
            with open(result_file, "w") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            print(f"Saved evaluation results to {result_file}")

        all_gen_results.append(results)

        # Compute or load accuracy
        acc = compute_accuracies_from_results(results)
        all_accuracies.append(acc)
        # print(acc)

    # Save all accuracies to a JSON file
    with open(accuracy_file, "w") as f:
        json.dump(all_accuracies, f, indent=2)
    print(f"Saved accuracies to {accuracy_file}")

    show_score_summary(all_accuracies)

if __name__ == "__main__":
    main()