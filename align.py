#!/home/dgao/anaconda3/envs/k2/bin/python

# 2021 Dongji Gao

import sys
import torch

sys.path.append("/export/c26/dgao/flexible_alignment/aligner")
sys.path.append("/export/c26/dgao/flexible_alignment/")
sys.path.append("/export/b14/dgao/snowfall")
sys.path.append("/export/b14/dgao/icefall")
sys.path.insert(0, "/home/dgao/anaconda3/envs/k2/lib/python3.8/site-packages/")


from aligner import FlexibleAligner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    help="pretrained model"
)
parser.add_argument(
    "dataset",
    type=str,
    help="huggingface dataset",
)
parser.add_argument(
    "text",
    type=str,
    help="text file",
)
parser.add_argument(
    "ses2spk",
    type=str,
    help="session to spk file",
)
parser.add_argument(
    "lang_dir",
    type=str,
    help="directory of lang",
)
parser.add_argument(
    "graph_dir",
    type=str,
    help="dirctory of graph",
)
parser.add_argument(
    "output_dir",
    type=str,
    help="output_dir",
)
parser.add_argument(
    "--xvector",
    action="store_true",
)
args = parser.parse_args()

model = args.model
dataset = args.dataset
text = args.text
ses2spk = args.ses2spk
lang_dir = args.lang_dir
graph_dir = args.graph_dir
output_dir = args.output_dir
use_xvector = args.xvector

f_aligner = FlexibleAligner(model, dataset, text, ses2spk, lang_dir, graph_dir, output_dir, use_xvector, "data/ntu_test_extend/xvector.json")
f_aligner.run()

print("Alignment done.")
