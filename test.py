"""
test.py - open-set evaluation part of mlaad-aasist_SSL.ipynb

Notebook cells covered: 45 (load best checkpoint + test + metrics), 46 (print), 47 (F1).

Usage:
    python test.py --data-root /path/to/processed_data_1 --checkpoint checkpoints/best_model_XLSR.pth
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import (ExpConfig, CONFIG, add_data_args, paths_from_args, build_datasets,
                    make_loader, PreEmphasis, Model, load_state_dict_flexible, progress)


def main():
    parser = argparse.ArgumentParser(description="Open-set test of XLS-R + AASIST")
    add_data_args(parser)
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model_XLSR.pth"))
    parser.add_argument("--results-json", default=None,
                        help="optional: also write the final metrics to this JSON file")
    args = parser.parse_args()

    exp_cfg = ExpConfig()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    # Train split is loaded only to rebuild the label mappings, exactly as the notebook did.
    paths = paths_from_args(args)
    ds = build_datasets(paths, exp_cfg, splits=("train", "test"))
    train_ds, test_ds = ds["train"], ds["test"]
    test_loader = make_loader(test_ds, exp_cfg, shuffle=False, num_workers=args.num_workers)

    pre = PreEmphasis(exp_cfg.pre_emphasis).to(DEVICE)

    # --------------------------------------------------
    # 1. Load the best checkpoint
    # --------------------------------------------------
    checkpoint = torch.load(
        args.checkpoint,
        map_location=DEVICE,
        weights_only=False
    )

    num_classes = checkpoint.get("num_classes", len(train_ds.model_to_idx))
    if "model_to_idx" in checkpoint and checkpoint["model_to_idx"] != train_ds.model_to_idx:
        raise RuntimeError("Label mapping in checkpoint differs from the train CSV - "
                           "wrong --data-root or wrong checkpoint?")

    model = Model(CONFIG["model_config"], num_classes=num_classes, device=DEVICE).to(DEVICE)
    load_state_dict_flexible(model, checkpoint["model_state_dict"])
    model.eval()

    prototypes = checkpoint["prototypes"].to(DEVICE)
    threshold = float(checkpoint["threshold"])

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    print(f"Open-set distance threshold: {threshold:.4f}")

    # --------------------------------------------------
    # 2. Test
    # --------------------------------------------------
    all_labels = []
    all_unknown_flags = []
    all_predicted_classes = []
    all_min_distances = []
    all_logits = []

    with torch.no_grad():
        pbar = progress(test_loader, desc="Testing", leave=True)

        for wav, model_label, language_label, model_unknown, language_unknown in pbar:
            wav = pre(wav.to(DEVICE))

            model_label = model_label.to(DEVICE)
            model_unknown = model_unknown.to(DEVICE).bool()

            # Expected output: embedding, logits
            embeddings, logits = model(wav)

            # Normalize embeddings before cosine-distance calculation
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Minimum cosine distance to any known-class prototype
            similarities = embeddings @ prototypes.T
            distances = 1.0 - similarities
            min_distances, nearest_classes = distances.min(dim=1)

            # Reject samples whose distance exceeds the dev threshold
            predicted_classes = nearest_classes.clone()
            predicted_classes[min_distances > threshold] = -1  # UNKNOWN

            all_labels.extend(model_label.cpu().numpy())
            all_unknown_flags.extend(model_unknown.cpu().numpy())
            all_predicted_classes.extend(predicted_classes.cpu().numpy())
            all_min_distances.extend(min_distances.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # Convert to NumPy arrays
    all_labels = np.asarray(all_labels)
    all_unknown_flags = np.asarray(all_unknown_flags, dtype=bool)
    all_predicted_classes = np.asarray(all_predicted_classes)
    all_min_distances = np.asarray(all_min_distances)

    # --------------------------------------------------
    # 3. Evaluate open-set performance
    # --------------------------------------------------
    known_mask = ~all_unknown_flags
    unknown_mask = all_unknown_flags

    # A known sample is correctly identified only if:
    # its predicted class matches its true class AND it was not rejected.
    known_correct = (
        known_mask
        & (all_predicted_classes == all_labels)
    )

    known_total = known_mask.sum()
    unknown_total = unknown_mask.sum()

    known_identification_rate = (
        known_correct.sum() / known_total
        if known_total > 0 else np.nan
    )

    known_acceptance_rate = (
        (all_predicted_classes[known_mask] != -1).mean()
        if known_total > 0 else np.nan
    )

    unknown_rejection_rate = (
        (all_predicted_classes[unknown_mask] == -1).mean()
        if unknown_total > 0 else np.nan
    )

    if known_total > 0 and unknown_total > 0:
        balanced_open_set_accuracy = (
            known_acceptance_rate + unknown_rejection_rate
        ) / 2
    else:
        balanced_open_set_accuracy = np.nan

    # Closed-set accuracy among accepted known samples
    accepted_known_mask = known_mask & (all_predicted_classes != -1)

    closed_set_accuracy = (
        (all_predicted_classes[accepted_known_mask]
         == all_labels[accepted_known_mask]).mean()
        if accepted_known_mask.sum() > 0 else np.nan
    )

    # --------------------------------------------------
    # 4. Print results  (cells 45/46)
    # --------------------------------------------------
    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test samples:                  {len(all_labels)}")
    print(f"Known samples:                 {known_total}")
    print(f"Unknown samples:               {unknown_total}")
    print(f"Known-source acceptance rate:  {known_acceptance_rate * 100:.2f}%")
    print(f"Unknown-source rejection rate: {unknown_rejection_rate * 100:.2f}%")
    print(f"Known-source identification:   {known_identification_rate * 100:.2f}%")
    print(f"Closed-set accuracy (accepted): {closed_set_accuracy * 100:.2f}%")
    print(f"Balanced open-set accuracy:    {balanced_open_set_accuracy * 100:.2f}%")
    print("========================================")

    # --------------------------------------------------
    # 5. Closed-set F1  (cell 47)
    # --------------------------------------------------
    # Closed-set evaluation: known samples only
    y_true = all_labels[known_mask]
    y_pred = all_predicted_classes[known_mask]

    # Exclude rejected samples (UNKNOWN = -1)
    accepted_mask = y_pred != -1

    y_true = y_true[accepted_mask]
    y_pred = y_pred[accepted_mask]

    # Macro and weighted F1
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Closed-set Macro F1-score:    {macro_f1 * 100:.2f}%")
    print(f"Closed-set Weighted F1-score: {weighted_f1 * 100:.2f}%")

    # (added) optional JSON copy of the numbers
    if args.results_json:
        results = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "threshold": threshold,
            "test_samples": int(len(all_labels)),
            "known_samples": int(known_total),
            "unknown_samples": int(unknown_total),
            "known_acceptance_rate": float(known_acceptance_rate),
            "unknown_rejection_rate": float(unknown_rejection_rate),
            "known_identification_rate": float(known_identification_rate),
            "closed_set_accuracy_accepted": float(closed_set_accuracy),
            "balanced_open_set_accuracy": float(balanced_open_set_accuracy),
            "closed_set_macro_f1": float(macro_f1),
            "closed_set_weighted_f1": float(weighted_f1),
        }
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print("Results written to", args.results_json)


if __name__ == "__main__":
    main()
