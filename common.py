"""
common.py - everything shared by train.py / test.py / flops.py / inspect_data.py

Converted from the notebook `mlaad-aasist_SSL.ipynb` (XLS-R 300M + AASIST back-end,
open-set source tracing on MLAAD). Class/function bodies are copied from the
notebook cells unchanged unless a comment says otherwise.

Notebook cell map:
  cell 5   -> ExpConfig, data paths (now command-line arguments)
  cell 9   -> AudioMetadataDataset
  cell 10  -> build_datasets()
  cell 12  -> build_loaders()
  cell 19  -> PreEmphasis
  cell 23  -> SSLModel, GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool,
              Residual_block, Model
  cell 27  -> CONFIG (only the parts that are actually used)
  cell 30  -> str_to_bool, set_seed
  cell 36  -> collect_embeddings, build_prototypes, minimum_cosine_distance,
              select_threshold
"""

import os
import sys
import random
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model


# ============================================================
# PROGRESS BAR
# In a terminal: normal tqdm. Inside a Slurm job (output goes to a file):
# refresh at most once a minute so the log file doesn't grow to gigabytes.
# ============================================================
def progress(iterable, **kwargs):
    if not sys.stderr.isatty():
        kwargs.setdefault("mininterval", 60)
    return tqdm(iterable, **kwargs)


# ============================================================
# CONFIG  (notebook cell 5)
# ============================================================
class ExpConfig:
    # Audio processing
    sample_rate = 16000
    pre_emphasis = 0.97
    train_duration_sec = 8
    test_duration_sec = 8

    # Model
    transformer_hidden = 660   # unused by this model (left over from Rawformer)

    # Training
    batch_size = 2
    lr = 8e-4
    epochs = 15


# notebook cell 27 - only "model_config" (passed to Model but ignored by it) and the
# cudnn toggles (used by set_seed) have any effect. The optimizer/scheduler/SWA built
# from "optim_config" in the notebook were never used by the training loop.
CONFIG = {
    "cudnn_deterministic_toggle": "True",
    "cudnn_benchmark_toggle": "False",
    "model_config": {
        "architecture": "AASIST",
        "nb_samp": 128000,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    },
}

SEED = 1234


def data_paths(data_root, path_train=None, path_dev=None, path_test=None,
               csv_train=None, csv_dev=None, csv_test=None):
    """Replaces notebook SysConfig. Same layout as the notebook:
        <data_root>/train/*.flac   <data_root>/train_metadata.csv
        <data_root>/dev/*.flac     <data_root>/dev_metadata.csv
        <data_root>/eval/*.flac    <data_root>/test_metadata.csv
    Any single path can be overridden."""
    j = os.path.join
    return {
        "path_train": path_train or j(data_root, "train"),
        "path_dev":   path_dev   or j(data_root, "dev"),
        "path_test":  path_test  or j(data_root, "eval"),
        "csv_train":  csv_train  or j(data_root, "train_metadata.csv"),
        "csv_dev":    csv_dev    or j(data_root, "dev_metadata.csv"),
        "csv_test":   csv_test   or j(data_root, "test_metadata.csv"),
    }


def add_data_args(parser):
    parser.add_argument("--data-root", required=True,
                        help="folder containing train/ dev/ eval/ and the 3 metadata CSVs")
    for k in ["path-train", "path-dev", "path-test", "csv-train", "csv-dev", "csv-test"]:
        parser.add_argument(f"--{k}", default=None, help="optional override")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (notebook used 0)")


def paths_from_args(args):
    return data_paths(args.data_root, args.path_train, args.path_dev, args.path_test,
                      args.csv_train, args.csv_dev, args.csv_test)


# ============================================================
# PRE-EMPHASIS  (notebook cell 19)
# ============================================================
class PreEmphasis(nn.Module):
    def __init__(self, pre_emphasis: float = 0.97):
        super().__init__()
        # Conv1D filter shape: (out_channels=1, in_channels=1, kernel_size=2)
        filt = torch.tensor([[-pre_emphasis, 1.0]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer("filter", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = x.unsqueeze(1)  # (B,1,T)
        x = F.pad(x, (1, 0), mode="reflect")
        x = F.conv1d(x, self.filter)
        return x.squeeze(1)  # (B,T)


# ============================================================
# DATASET  (notebook cell 9)
# ============================================================
class AudioMetadataDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir,
        csv_path,
        sample_rate=16000,
        duration_sec=2,
        model_to_idx=None,
        language_to_idx=None
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        # ----------------------------------------------------
        # Read metadata
        # ----------------------------------------------------
        self.df = pd.read_csv(csv_path)

        # ----------------------------------------------------
        # TRAIN creates the known mappings
        # DEV/TEST receive the TRAIN mappings
        # ----------------------------------------------------
        if model_to_idx is None:

            models = sorted(
                self.df["model"].dropna().unique()
            )

            self.model_to_idx = {
                model: idx
                for idx, model in enumerate(models)
            }

        else:
            self.model_to_idx = model_to_idx

        if language_to_idx is None:

            languages = sorted(
                self.df["language"].dropna().unique()
            )

            self.language_to_idx = {
                lang: idx
                for idx, lang in enumerate(languages)
            }

        else:
            self.language_to_idx = language_to_idx

        # ----------------------------------------------------
        # Explicit UNKNOWN IDs
        # ----------------------------------------------------

        self.UNKNOWN_MODEL = len(self.model_to_idx)
        self.UNKNOWN_LANGUAGE = len(self.language_to_idx)

        # ----------------------------------------------------
        # Reverse mappings
        # ----------------------------------------------------

        self.idx_to_model = {
            idx: model
            for model, idx in self.model_to_idx.items()
        }

        self.idx_to_model[self.UNKNOWN_MODEL] = "UNKNOWN_MODEL"

        self.idx_to_language = {
            idx: lang
            for lang, idx in self.language_to_idx.items()
        }

        self.idx_to_language[self.UNKNOWN_LANGUAGE] = "UNKNOWN_LANGUAGE"

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print(f"\n📁 Loaded {len(self.df)} files")
        print(f"   Root: {root_dir}")

        print(
            f"   Known models: "
            f"{len(self.model_to_idx)}"
        )

        print(
            f"   Known languages: "
            f"{len(self.language_to_idx)}"
        )

        print(
            f"   UNKNOWN_MODEL ID: "
            f"{self.UNKNOWN_MODEL}"
        )

        print(
            f"   UNKNOWN_LANGUAGE ID: "
            f"{self.UNKNOWN_LANGUAGE}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ====================================================
        # AUDIO
        # ====================================================

        filename = row["filename"]

        path = os.path.join(
            self.root_dir,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        wav_np, sr = sf.read(path)

        # Stereo -> mono
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        wav = torch.tensor(
            wav_np,
            dtype=torch.float32
        ).unsqueeze(0)

        # Resample
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.sample_rate
            )

        # Crop / pad
        num_samples = int(
            self.sample_rate *
            self.duration_sec
        )

        if wav.size(1) > num_samples:

            start = random.randint(
                0,
                wav.size(1) - num_samples
            )

            wav = wav[
                :,
                start:start + num_samples
            ]

        elif wav.size(1) < num_samples:

            wav = F.pad(
                wav,
                (0, num_samples - wav.size(1))
            )

        # ====================================================
        # MODEL LABEL
        # ====================================================

        model_name = row["model"]

        if model_name in self.model_to_idx:

            model_label = self.model_to_idx[
                model_name
            ]

            model_is_unknown = 0

        else:

            model_label = self.UNKNOWN_MODEL
            model_is_unknown = 1

        # ====================================================
        # LANGUAGE LABEL
        # ====================================================

        language_name = row["language"]

        if language_name in self.language_to_idx:

            language_label = self.language_to_idx[
                language_name
            ]

            language_is_unknown = 0

        else:

            language_label = self.UNKNOWN_LANGUAGE
            language_is_unknown = 1

        # ====================================================
        # RETURN
        # ====================================================

        return (
            wav.squeeze(0),

            torch.tensor(
                model_label,
                dtype=torch.long
            ),

            torch.tensor(
                language_label,
                dtype=torch.long
            ),

            torch.tensor(
                model_is_unknown,
                dtype=torch.long
            ),

            torch.tensor(
                language_is_unknown,
                dtype=torch.long
            )
        )


def build_datasets(paths, exp_cfg, splits=("train", "dev", "test")):
    """notebook cell 10. TRAIN creates the label mappings; DEV/TEST reuse them."""
    out = {}
    train_ds = AudioMetadataDataset(
        root_dir=paths["path_train"],
        csv_path=paths["csv_train"],
        sample_rate=exp_cfg.sample_rate,
        duration_sec=exp_cfg.train_duration_sec
    )
    out["train"] = train_ds
    model_to_idx = train_ds.model_to_idx
    language_to_idx = train_ds.language_to_idx

    if "dev" in splits:
        out["dev"] = AudioMetadataDataset(
            root_dir=paths["path_dev"],
            csv_path=paths["csv_dev"],
            sample_rate=exp_cfg.sample_rate,
            duration_sec=exp_cfg.test_duration_sec,
            model_to_idx=model_to_idx,
            language_to_idx=language_to_idx
        )
    if "test" in splits:
        out["test"] = AudioMetadataDataset(
            root_dir=paths["path_test"],
            csv_path=paths["csv_test"],
            sample_rate=exp_cfg.sample_rate,
            duration_sec=exp_cfg.test_duration_sec,
            model_to_idx=model_to_idx,
            language_to_idx=language_to_idx
        )
    return out


def make_loader(ds, exp_cfg, shuffle, num_workers=0):
    """notebook cell 12 (train: shuffle=True, dev/test: shuffle=False)."""
    return torch.utils.data.DataLoader(
        ds,
        batch_size=exp_cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ============================================================
# MODEL: XLS-R 300M front-end + AASIST back-end  (notebook cell 23)
# ============================================================
class SSLModel(nn.Module):

    def __init__(self, device):
        super(SSLModel, self).__init__()

        self.device = device

        print("[INFO] Loading XLS-R 300M...")

        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            revision="bff87cb13ec13ef9b3823739b2a721e3bf08dd76",
            use_safetensors=True
        )

        self.out_dim = 1024
        self.model.to(device)

        print("[SUCCESS] XLS-R loaded without Fairseq!")

    def extract_feat(self, input_data):

        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        input_data = input_data.to(self.device)

        emb = self.model(
            input_values=input_data
        ).last_hidden_state

        return emb


#---------AASIST back-end------------------------#
''' Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans. 
    AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. 
    In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        #print('num_type1',num_type1)
        #print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        #print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        #print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('Concat x1 and x2',x.shape)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            #print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        #print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        #print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        #print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        #print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        #print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h




class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)
        return out


class Model(nn.Module):
    def __init__(self, d_args, num_classes:int, device="cuda"):
        super().__init__()
        # NOTE: notebook read the global `device` here ("cuda"/"cpu"); now passed in.
        self.device = device
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], num_classes)

    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return last_hidden, output


# ============================================================
# SEEDING  (notebook cell 30)
# ============================================================
def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0)."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def set_seed(seed, config=None):
    """Set initial seed for reproduction"""
    if config is None:
        raise ValueError("config should not be None")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])


def load_state_dict_flexible(model, state_dict):
    """Loads a checkpoint whether or not it was saved from a DataParallel model."""
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


# ============================================================
# OPEN-SET HELPERS  (notebook cell 36)
# ============================================================
@torch.no_grad()
def collect_embeddings(model, loader, device, pre, desc="Extracting embeddings"):
    model.eval()

    embeddings = []
    labels = []
    is_unknown = []

    pbar = progress(loader, desc=desc, leave=True)

    for wav, model_label, language_label, model_unknown, language_unknown in pbar:
        wav = pre(wav.to(device))

        emb, logits = model(wav)

        embeddings.append(
            F.normalize(emb, p=2, dim=1).cpu()
        )
        labels.append(model_label.cpu())
        is_unknown.append(model_unknown.cpu())

    return (
        torch.cat(embeddings),
        torch.cat(labels),
        torch.cat(is_unknown).bool()
    )


def build_prototypes(embeddings, labels, is_unknown, num_classes):
    prototypes = []

    for class_id in range(num_classes):
        mask = (labels == class_id) & (~is_unknown)

        if not mask.any():
            raise ValueError(
                f"No training embeddings for class {class_id}"
            )

        # Mean embedding, then normalize.
        proto = embeddings[mask].mean(dim=0)
        proto = F.normalize(proto.unsqueeze(0), p=2, dim=1).squeeze(0)

        prototypes.append(proto)

    return torch.stack(prototypes)


def minimum_cosine_distance(embeddings, prototypes):
    # Both inputs are normalized.
    # Distance = 1 - cosine similarity.
    similarities = embeddings @ prototypes.T
    distances = 1.0 - similarities

    min_distances, nearest_class = distances.min(dim=1)

    return min_distances, nearest_class


def select_threshold(dev_distances, dev_is_unknown):
    """
    Select threshold using balanced accuracy:
      Known samples: accept if distance <= threshold
      Unknown samples: reject if distance > threshold
    """
    known_dist = dev_distances[~dev_is_unknown]
    unknown_dist = dev_distances[dev_is_unknown]

    if len(known_dist) == 0 or len(unknown_dist) == 0:
        raise ValueError(
            "Threshold selection requires BOTH known and unknown "
            "development samples."
        )

    # Test thresholds between observed distances.
    values = np.sort(dev_distances.numpy())
    thresholds = np.concatenate([
        [values[0] - 1e-7],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + 1e-7]
    ])

    best_threshold = None
    best_score = -1.0

    for threshold in thresholds:
        known_accept_rate = (
            known_dist <= threshold
        ).float().mean().item()

        unknown_reject_rate = (
            unknown_dist > threshold
        ).float().mean().item()

        balanced_accuracy = (
            known_accept_rate + unknown_reject_rate
        ) / 2

        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = float(threshold)

    return best_threshold, best_score
