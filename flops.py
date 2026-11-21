"""
flops.py - parameter count + FLOPs part of mlaad-aasist_SSL.ipynb

Notebook cells covered: 50 (params / FLOPs / torchinfo table), 52 (GFLOPs per second of audio).
FLOPs don't depend on trained weights, so no checkpoint or dataset is needed.

Usage:
    python flops.py                 # 24 classes, as in the notebook
    python flops.py --num-classes 24 --seconds 4
"""

import argparse

import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

from common import ExpConfig, CONFIG, PreEmphasis, Model


# --------------------------------------------------------------
# 1. Parameter count (trainable + non-trainable) + size in MiB
# --------------------------------------------------------------
def print_model_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    size_mb   = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    print("\n" + "="*60)
    print("MODEL PARAMETER SUMMARY")
    print("="*60)
    print(f"{'Trainable params':<25}: {trainable:,}")
    print(f"{'Total params'    :<25}: {total:,}")
    print(f"{'Model size (MiB)':<25}: {size_mb:.2f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Params / FLOPs of XLS-R + AASIST")
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="dummy input length; the notebook fell back to 4.0 s")
    args = parser.parse_args()

    exp_cfg = ExpConfig()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    pre = PreEmphasis(exp_cfg.pre_emphasis).to(DEVICE)
    model = Model(CONFIG["model_config"], num_classes=args.num_classes, device=DEVICE).to(DEVICE)
    model.eval()   # in the notebook the model was in eval mode here (after the test cell)

    print_model_params(model)

    # --------------------------------------------------------------
    # 2. FLOPs / MACs
    # --------------------------------------------------------------
    max_samples = int(exp_cfg.sample_rate * args.seconds)
    dummy_wav = torch.randn(1, max_samples, device=DEVICE)     # (B, T)

    # Apply pre-emphasis as in training/validation
    if pre is not None:
        dummy_wav = pre(dummy_wav)

    # ---- fvcore ----
    flops = FlopCountAnalysis(model, dummy_wav)
    macs  = flops.total()                # fvcore counts one "flop" per multiply-add
    flops_2 = macs * 2                   # FLOPs = 2 x MACs (standard convention)

    print("\n" + "="*60)
    print("FLOPs / MACs (per forward pass)")
    print("="*60)
    print(f"{'Input shape'   :<25}: {list(dummy_wav.shape)}")
    print(f"{'MACs'          :<25}: {macs/1e9:.3f} G")
    print(f"{'FLOPs'         :<25}: {flops_2/1e9:.3f} G")
    print("="*60 + "\n")

    # ---- torchinfo (layer table) ----
    print("Detailed layer-wise breakdown (torchinfo):")
    print(summary(model,
                  input_data=dummy_wav,
                  col_names=["input_size", "output_size", "num_params", "mult_adds"],
                  depth=4,
                  verbose=0))

    # --------------------------------------------------------------
    # 3. FLOPs per second of audio  (cell 52)
    # --------------------------------------------------------------
    seconds = max_samples / exp_cfg.sample_rate
    print(f"GFLOPs per second of audio : {flops_2/1e9/seconds:.3f}")


if __name__ == "__main__":
    main()
