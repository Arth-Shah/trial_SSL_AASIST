# XLS-R + AASIST (MLAAD open-set source tracing) - terminal / Slurm version

Converted from `mlaad-aasist_SSL.ipynb`. The model, data pipeline, training loop,
threshold selection and test metrics are the notebook's code, copied cell-for-cell
(checked automatically: every class/function is identical except the small edits listed below).

## Files

| File | What it is | Notebook cells |
|---|---|---|
| `common.py` | config, dataset, pre-emphasis, XLS-R + AASIST model, open-set helpers | 5, 9, 10, 12, 19, 23, 27, 30, 36 |
| `inspect_data.py` | checks paths/CSVs, label maps, unseen labels, open-set counts (no GPU) | 10, 13, 14, 15, 16 |
| `download_model.py` | downloads XLS-R 300M once (run on master node) | - |
| `train.py` | training + per-epoch threshold selection + best checkpoint | 1, 13, 20, 24, 26, 32, 35, 38, 40 |
| `test.py` | open-set test + closed-set F1 | 45, 46, 47 |
| `flops.py` | parameter count, FLOPs, torchinfo table, GFLOPs per second | 50, 52 |
| `train_job.sh`, `test_job.sh` | Slurm scripts for the DAU cluster | - |

## Step-by-step on the cluster

**0. Your faculty account must be unblocked first** (`f200791001` showed BLOCKED / negative
tokens at login). Until the admin tops it up, `sbatch` will reject these jobs.

**1. Copy the code and data** (MobaXterm left panel, or `scp` from your PC):
```
~/mlaad_aasist_ssl/            <- these files
~/data/processed_data_1/
    train/*.flac   dev/*.flac   eval/*.flac
    train_metadata.csv   dev_metadata.csv   test_metadata.csv
```
Tip: zip `processed_data_1` on Windows, upload the single zip, then `unzip` on the cluster -
56,891 small files upload far faster as one archive. Avoid the `'` in `ICASSP'27` in Linux paths.

**2. Create the Python environment** (once, on the master node):
```bash
module load anaconda3-2025-12-2
source /apps/compilers/anaconda3-2025-12-2/etc/profile.d/conda.sh
conda create -n aasist python=3.11 -y
conda activate aasist
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
cd ~/mlaad_aasist_ssl
pip install -r requirements.txt
```
Your notebook used torch 2.5.1+cu121. That build does **not** support the RTX PRO 6000
(Blackwell) GPUs, so a CUDA 12.8 build is required. If the master node has no internet,
ask the HPC admin to install these packages.

**3. Download XLS-R once and check the data** (master node, no GPU needed):
```bash
python download_model.py
python inspect_data.py --data-root ~/data/processed_data_1
```
You should see the same numbers as the notebook: 11100 / 12000 / 33791 files, 24 models, 8 languages.

**4. Train:**
```bash
mkdir -p logs
sbatch train_job.sh
squeue -u $USER
tail -f logs/train-<JOBID>.out
```
If the job hits its 24 h limit, run `sbatch train_job.sh` again. It continues from
`checkpoints/last_model_XLSR.pth`. To start from scratch, delete `checkpoints/` first.

**5. Test:**
```bash
sbatch test_job.sh
cat logs/test-<JOBID>.out
```

Running directly (e.g. inside an interactive GPU session):
```bash
python train.py --data-root ~/data/processed_data_1
python test.py  --data-root ~/data/processed_data_1 --checkpoint checkpoints/best_model_XLSR.pth
python flops.py
```
All options: `python train.py --help` (`--epochs`, `--save-path`, `--resume`, `--num-workers`,
and per-folder/CSV path overrides).

## What changed compared with the notebook

Edits to the code itself (the only ones):
- Windows paths -> `--data-root` argument; checkpoint goes to `checkpoints/best_model_XLSR.pth` (same name as cell 38).
- `Model(...)` takes `device=` instead of reading the global variable `device`.
- `collect_embeddings(...)` takes `pre` instead of reading the global variable `pre`.
- `tqdm` progress bars refresh once a minute when writing to a Slurm log (normal in a terminal).

Additions (don't change training results):
- `checkpoints/last_model_XLSR.pth` saved every epoch + `--resume`, replacing commented-out cell 39.
- `--results-json` in `test.py`; `--num-workers` (notebook used 0; the job scripts use 4 for speed,
  which only changes which random 8 s crop each file gets).

Removed because they never affected the result:
- `WaveformAugmentation` (cell 7) - defined but never used.
- Cell 21 - the original AASIST, fully commented out.
- Adam + cosine scheduler + SWA built from `config["optim_config"]` (cells 27, 30, 35) - created but
  never stepped; training only uses `opt = Adam(lr=8e-4)`. So `torchcontrib` is no longer needed.
- `DataParallel` wrapper - unused with 1 GPU, and it would crash with 2 GPUs because SSLModel
  sends every input to GPU 0.
- Cell 44 (old binary EER/t-DCF test, commented out) and all `pip install` cells.

## Things in the notebook worth knowing (kept exactly as they were)
1. `Residual_block.forward` computes `bn1 -> selu` and then throws it away (`out = self.conv1(x)`),
   so `bn1` is never used. This comes from the original SSL-AASIST code.
2. The whole 300M-parameter XLS-R is fine-tuned with Adam at lr 8e-4 and batch size 2. That learning
   rate is far higher than what is usually used to fine-tune wav2vec2/XLS-R (around 1e-6 to 1e-5).
   Your log shows 16% train accuracy after epoch 1. Change `ExpConfig.lr` in `common.py` if you want.
3. Dev and test files longer than 8 s are cropped at a **random** position, so test numbers can
   change slightly between runs.
4. Every epoch re-extracts embeddings for the full training set to build prototypes. That is about
   20-25% of each epoch's time.
5. The FLOPs output saved in the notebook (301,408 params, `CONV` layer) belongs to an older
   plain-AASIST model, not this XLS-R model. `flops.py` gives the real numbers.
6. The saved test results (epoch-14 checkpoint, 70.85% balanced open-set accuracy) came from an
   earlier run. The last training run saved in the notebook was stopped during epoch 1.
