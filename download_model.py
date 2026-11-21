"""
download_model.py - fetch the pretrained XLS-R 300M weights ONCE, on the login (master) node.

The GPU nodes may not have internet access. Run this on the master node first; the
weights (~1.27 GB) go into the Hugging Face cache in your home folder (~/.cache/huggingface),
which the GPU nodes see too. The job scripts then set HF_HUB_OFFLINE=1.

Usage:
    python download_model.py
"""

from transformers import Wav2Vec2Model

# Same model + pinned revision as SSLModel in the notebook (cell 23)
Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    revision="bff87cb13ec13ef9b3823739b2a721e3bf08dd76",
    use_safetensors=True,
)
print("XLS-R 300M is cached. You can now submit the Slurm jobs.")
