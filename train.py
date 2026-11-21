"""
train.py - training part of mlaad-aasist_SSL.ipynb

Notebook cells covered: 1, 10, 12, 13, 20, 24, 26, 32, 35, 38, 40.

Usage:
    python train.py --data-root /path/to/processed_data_1
    python train.py --data-root /path/to/processed_data_1 --resume checkpoints/last_model_XLSR.pth
"""

import os
import argparse

import torch
import torch.nn as nn

from common import (ExpConfig, CONFIG, SEED, add_data_args, paths_from_args,
                    build_datasets, make_loader, PreEmphasis, Model, set_seed,
                    load_state_dict_flexible, collect_embeddings, build_prototypes,
                    minimum_cosine_distance, select_threshold, progress)


def main():
    parser = argparse.ArgumentParser(description="Train XLS-R + AASIST (open-set source tracing)")
    add_data_args(parser)
    parser.add_argument("--save-path", default=os.path.join("checkpoints", "best_model_XLSR.pth"),
                        help="best checkpoint (lowest known-class dev loss). Notebook cell 38.")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"override ExpConfig.epochs (default {ExpConfig.epochs})")
    parser.add_argument("--resume", default=None,
                        help="path to a last_*.pth checkpoint to continue from")
    args = parser.parse_args()

    exp_cfg = ExpConfig()
    if args.epochs is not None:
        exp_cfg.epochs = args.epochs

    # ---------------- notebook cell 1: CUDA check ----------------
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"[{i}] {torch.cuda.get_device_name(i)}")
    print(torch.__version__)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- notebook cells 10, 12: data ----------------
    paths = paths_from_args(args)
    ds = build_datasets(paths, exp_cfg, splits=("train", "dev"))   # test set not needed here
    train_ds, val_ds = ds["train"], ds["dev"]
    train_loader = make_loader(train_ds, exp_cfg, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, exp_cfg, shuffle=False, num_workers=args.num_workers)

    # ---------------- notebook cell 13: label mappings ----------------
    print("\n==============================")
    print("MODEL LABEL MAPPING")
    print("==============================")
    for name, idx in train_ds.model_to_idx.items():
        print(f"{idx}: {name}")
    print("\n==============================")
    print("LANGUAGE LABEL MAPPING")
    print("==============================")
    for name, idx in train_ds.language_to_idx.items():
        print(f"{idx}: {name}")

    # ---------------- notebook cell 20: pre-emphasis ----------------
    pre = PreEmphasis(exp_cfg.pre_emphasis).to(DEVICE)

    # ---------------- notebook cells 24, 26 ----------------
    num_classes = len(train_ds.model_to_idx)
    criterion = nn.CrossEntropyLoss()
    print("Known source classes:", num_classes)
    print("Training label mapping:", train_ds.model_to_idx)

    # ---------------- notebook cell 32: seed + model ----------------
    set_seed(SEED, CONFIG)
    model = Model(CONFIG["model_config"], num_classes=num_classes, device=DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # NOTE: the notebook wrapped the model in DataParallel when >1 GPU was visible.
    # Left out on purpose: SSLModel moves inputs to "cuda" (= GPU 0), which breaks
    # DataParallel replicas on other GPUs. With 1 GPU (as in the notebook) nothing changes.
    nb_params = sum([param.view(-1).size(0) for param in model.parameters()])
    print(f"No. model params: {nb_params}")
    print(f"Device: {device}")
    if str(device) == "cpu":
        print("Warning: Running on CPU, which may be slow!")

    # ---------------- notebook cell 35: optimizer ----------------
    # The notebook also built Adam + cosine scheduler + SWA from config["optim_config"],
    # but the training loop never used them - it only ever calls `opt`. Omitted.
    opt = torch.optim.Adam(model.parameters(), lr=exp_cfg.lr)

    # ---------------- notebook cell 38: checkpoint path ----------------
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    last_path = os.path.join(os.path.dirname(save_path),
                             "last_" + os.path.basename(save_path).replace("best_", "", 1))
    print("Checkpoint will be saved to:", save_path)

    best_val_loss = float("inf")
    best_threshold = None
    start_epoch = 1

    # ---------------- (added) resume - replaces the commented-out notebook cell 39 ----------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        load_state_dict_flexible(model, ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        best_threshold = ckpt.get("best_threshold", ckpt.get("threshold"))
        print(f"Resumed from {args.resume}: continuing at epoch {start_epoch}, "
              f"best dev loss so far {best_val_loss:.4f}")

    # ---------------- notebook cell 40: training loop ----------------
    for epoch in range(start_epoch, exp_cfg.epochs + 1):

        # ========================================================
        # TRAIN
        # ========================================================
        model.train()

        total_loss = 0.0
        total_samples = 0
        correct = 0

        pbar = progress(
            train_loader,
            desc=f"Epoch {epoch}/{exp_cfg.epochs} [Train]"
        )

        for wav, model_label, language_label, model_unknown, language_unknown in pbar:

            wav = pre(wav.to(DEVICE))
            model_label = model_label.to(DEVICE)
            model_unknown = model_unknown.to(DEVICE).bool()

            # Only known training classes contribute to CE loss.
            known_mask = ~model_unknown

            if not known_mask.any():
                continue

            opt.zero_grad()

            _, logits = model(wav)

            loss = criterion(
                logits[known_mask],
                model_label[known_mask]
            )

            loss.backward()
            opt.step()

            bs = int(known_mask.sum().item())
            total_loss += loss.item() * bs
            total_samples += bs

            predictions = logits[known_mask].argmax(dim=1)
            correct += (
                predictions == model_label[known_mask]
            ).sum().item()

            pbar.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{100 * correct / max(total_samples, 1):.2f}%"
            )

        avg_train_loss = total_loss / max(total_samples, 1)
        train_acc = correct / max(total_samples, 1)

        # ========================================================
        # DEVELOPMENT: MULTI-CLASS LOSS
        # ========================================================
        model.eval()
    
        val_loss_sum = 0.0
        val_samples = 0
        val_correct = 0
    
        with torch.no_grad():
            val_pbar = progress(
                val_loader,
                desc=f"Epoch {epoch}/{exp_cfg.epochs} [Validation]",
                leave=True
            )
    
            for wav, model_label, language_label, model_unknown, language_unknown in val_pbar:
    
                wav = pre(wav.to(DEVICE))
                model_label = model_label.to(DEVICE)
                model_unknown = model_unknown.to(DEVICE).bool()
    
                _, logits = model(wav)
                known_mask = ~model_unknown
    
                if known_mask.any():
                    loss = criterion(
                        logits[known_mask],
                        model_label[known_mask]
                    )
    
                    n = int(known_mask.sum().item())
                    val_loss_sum += loss.item() * n
                    val_samples += n
    
                    pred = logits[known_mask].argmax(dim=1)
                    val_correct += (
                        pred == model_label[known_mask]
                    ).sum().item()
    
                val_pbar.set_postfix(
                    loss=f"{val_loss_sum / max(val_samples, 1):.4f}",
                    acc=f"{100 * val_correct / max(val_samples, 1):.2f}%"
                )
    
        avg_val_loss = val_loss_sum / max(val_samples, 1)
        val_known_acc = val_correct / max(val_samples, 1)

        # ========================================================
        # OPEN-SET THRESHOLD SELECTION
        # ========================================================
    
        train_emb, train_labels, train_unknown = collect_embeddings(
            model,
            train_loader,
            DEVICE,
            pre,
            desc=f"Epoch {epoch}/{exp_cfg.epochs} [Train Embeddings]"
        )
    
        prototypes = build_prototypes(
            train_emb,
            train_labels,
            train_unknown,
            num_classes
        )
    
        dev_emb, dev_labels, dev_unknown = collect_embeddings(
            model,
            val_loader,
            DEVICE,
            pre,
            desc=f"Epoch {epoch}/{exp_cfg.epochs} [Dev Embeddings]"
        )
    
        dev_distances, dev_nearest = minimum_cosine_distance(
            dev_emb,
            prototypes
        )
    
        threshold, dev_balanced_acc = select_threshold(
            dev_distances,
            dev_unknown
        )

        # Metrics at the selected development threshold.
        accepted_as_known = dev_distances <= threshold

        known_mask = ~dev_unknown
        unknown_mask = dev_unknown

        known_accept_rate = accepted_as_known[known_mask].float().mean().item()
        unknown_reject_rate = (~accepted_as_known[unknown_mask]).float().mean().item()

        # Closed-set source accuracy, conditional on being accepted.
        correctly_identified = (
            (dev_nearest == dev_labels) & accepted_as_known & known_mask
        )
        known_total = int(known_mask.sum().item())
        unknown_total = int(unknown_mask.sum().item())

        # ========================================================
        # REPORT
        # ========================================================
        print(f"\nEpoch {epoch} Summary")
        print(f"Train Loss:                  {avg_train_loss:.4f}")
        print(f"Train Known-Source Accuracy: {train_acc * 100:.2f}%")
        print(f"Dev Known-Source Loss:       {avg_val_loss:.4f}")
        print(f"Dev Known-Source Accuracy:  {val_known_acc * 100:.2f}%")
        print(f"Selected Distance Threshold:{threshold:.4f}")
        print(f"Dev Balanced Open-Set Acc:   {dev_balanced_acc * 100:.2f}%")
        print(f"Known Acceptance Rate:       {known_accept_rate * 100:.2f}%")
        print(f"Unknown Rejection Rate:      {unknown_reject_rate * 100:.2f}%")
        print(
            "Correct Known ID among All Known: "
            f"{correctly_identified.sum().item() / max(known_total, 1) * 100:.2f}%"
        )

        # ========================================================
        # SAVE BEST MODEL + THRESHOLD
        # ========================================================
        # Choose your checkpoint criterion in advance.
        # Here, we select the model with the lowest known-class dev loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_threshold = threshold

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "num_classes": num_classes,
                "model_to_idx": train_ds.model_to_idx,
                "threshold": best_threshold,
                "prototypes": prototypes,
                "val_loss": best_val_loss,
                "dev_balanced_open_set_acc": dev_balanced_acc,
            }, save_path)

            print(f"Saved best checkpoint: {save_path}")
            print(f"Saved development threshold: {best_threshold:.4f}")

        # --------------------------------------------------------
        # (added) SAVE "LAST" CHECKPOINT EVERY EPOCH -> lets you --resume
        # if the Slurm job hits its time limit. Does not change training.
        # --------------------------------------------------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "num_classes": num_classes,
            "model_to_idx": train_ds.model_to_idx,
            "best_val_loss": best_val_loss,
            "best_threshold": best_threshold,
        }, last_path)
        print(f"Saved resume checkpoint: {last_path}")

        print("-" * 65)

    print("Training finished. Best dev loss:", best_val_loss)


if __name__ == "__main__":
    main()
