#!/usr/bin/env python3

# 2021 Dongji Gao

import sys
sys.path.append("/export/b14/dgao/flexible_alignment/aligner")
sys.path.append("/export/b14/dgao/flexible_alignment/")
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
args = parser.parse_args()

model = args.model
dataset = args.dataset
text = args.text
ses2spk = args.ses2spk
lang_dir = args.lang_dir
graph_dir = args.graph_dir
output_dir = args.output_dir

f_aligner = FlexibleAligner(model, dataset, text, ses2spk, lang_dir, graph_dir, output_dir)
f_aligner.run()

print("Alignment done.")
