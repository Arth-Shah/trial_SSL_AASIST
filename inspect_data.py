"""
inspect_data.py - dataset checks from mlaad-aasist_SSL.ipynb (no GPU needed)

Notebook cells covered: 10 (load splits), 13 (label mappings), 14 (one dev batch),
15 (unseen labels in dev/eval), 16 (open-set statistics).
Run this first on the cluster to confirm your paths and CSVs are right.

Usage:
    python inspect_data.py --data-root /path/to/processed_data_1
"""

import argparse

from common import ExpConfig, add_data_args, paths_from_args, build_datasets, make_loader


def inspect_unseen_labels(name, dataset, model_to_idx, language_to_idx):
    df = dataset.df

    # Normalize to strings to make comparison consistent.
    models = df["model"].astype(str).str.strip()
    languages = df["language"].astype(str).str.strip()

    unseen_models = sorted(set(models) - set(model_to_idx))
    unseen_languages = sorted(set(languages) - set(language_to_idx))

    print(f"\n{name}")
    print("Unique model labels in split:", models.nunique())
    print("Unique language labels in split:", languages.nunique())
    print("Unseen model labels:", len(unseen_models))
    print("Unseen language labels:", len(unseen_languages))

    print("\nUnseen models:")
    for model in unseen_models:
        print(" ", model)

    print("\nUnseen languages:")
    for language in unseen_languages:
        print(" ", language)


def print_open_set_stats(name, dataset):
    model_unknown = sum(
        dataset.df["model"].map(dataset.model_to_idx).isna()
    )

    language_unknown = sum(
        dataset.df["language"].map(dataset.language_to_idx).isna()
    )

    both_unknown = sum(
        (~dataset.df["model"].isin(dataset.model_to_idx))
        & (~dataset.df["language"].isin(dataset.language_to_idx))
    )

    print(f"\n{name}")
    print(f"Total samples:       {len(dataset)}")
    print(f"Unknown model:       {model_unknown}")
    print(f"Unknown language:    {language_unknown}")
    print(f"Both unknown:        {both_unknown}")


def main():
    parser = argparse.ArgumentParser(description="Inspect MLAAD metadata splits")
    add_data_args(parser)
    args = parser.parse_args()

    exp_cfg = ExpConfig()
    ds = build_datasets(paths_from_args(args), exp_cfg)
    train_ds, val_ds, test_ds = ds["train"], ds["dev"], ds["test"]
    model_to_idx = train_ds.model_to_idx
    language_to_idx = train_ds.language_to_idx

    # cell 13
    print("\n==============================")
    print("MODEL LABEL MAPPING")
    print("==============================")
    for name, idx in model_to_idx.items():
        print(f"{idx}: {name}")
    print("\n==============================")
    print("LANGUAGE LABEL MAPPING")
    print("==============================")
    for name, idx in language_to_idx.items():
        print(f"{idx}: {name}")

    # cell 14 - also proves the audio files can actually be read
    val_loader = make_loader(val_ds, exp_cfg, shuffle=False, num_workers=args.num_workers)
    wav, model_label, language_label, model_is_unknown, language_is_unknown = next(iter(val_loader))
    print("\nAudio shape:             ", wav.shape)
    print("Model labels shape:      ", model_label.shape)
    print("Language labels shape:   ", language_label.shape)
    print("Unknown model flags:     ", model_is_unknown.shape)
    print("Unknown language flags:  ", language_is_unknown.shape)
    print("First model label:       ", model_label[0].item())
    print("First language label:    ", language_label[0].item())
    print("First model unknown?:    ", model_is_unknown[0].item())
    print("First language unknown?: ", language_is_unknown[0].item())

    # cell 15
    inspect_unseen_labels("DEV", val_ds, model_to_idx, language_to_idx)
    inspect_unseen_labels("EVAL", test_ds, model_to_idx, language_to_idx)

    # cell 16
    print_open_set_stats("TRAIN", train_ds)
    print_open_set_stats("DEV", val_ds)
    print_open_set_stats("EVAL", test_ds)


if __name__ == "__main__":
    main()
