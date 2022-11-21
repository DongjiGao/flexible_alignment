#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  Johns Hopkins University   (author: Dongji Gao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script takes as input lang_dir and generates HLG from

    - H, the ctc topology, built from tokens contained in lang_dir/lexicon.txt
    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated HLG is saved in $lang_dir/HLG.pt
"""
import argparse
import logging
from pathlib import Path

import k2
import torch

from icefall.lexicon import Lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    return parser.parse_args()


def compile_HLG(lang_dir: str) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone.

    Return:
      An List of FSAs representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))
    HLG_list = []

    logging.info("Loading G.fst.txt")
    with open(lang_dir / "G.fst.txt") as f:
        g_fsts = f.read().strip().split("\n\n")
    
    for g_fst in g_fsts:
        G = k2.Fsa.from_str(g_fst, acceptor=False)

        first_token_disambig_id = lexicon.token_table["#0"]
        first_word_disambig_id = lexicon.word_table["#0"]

        L = k2.arc_sort(L)
        G = k2.arc_sort(G)

        logging.info("Intersecting L and G")
        LG = k2.compose(L, G)
        logging.info(f"LG shape: {LG.shape}")

        logging.info("Connecting LG")
        LG = k2.connect(LG)
        logging.info(f"LG shape after k2.connect: {LG.shape}")

        logging.info(type(LG.aux_labels))
        logging.info("Determinizing LG")

        LG = k2.determinize(LG)
        logging.info(type(LG.aux_labels))

        logging.info("Connecting LG after k2.determinize")
        LG = k2.connect(LG)

        logging.info("Removing disambiguation symbols on LG")

        LG.labels[LG.labels >= first_token_disambig_id] = 0

        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

        LG = k2.remove_epsilon(LG)
        logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)

        logging.info("Arc sorting LG")
        LG = k2.arc_sort(LG)

        logging.info("Composing H and LG")
        # CAUTION: The name of the inner_labels is fixed
        # to `tokens`. If you want to change it, please
        # also change other places in icefall that are using
        # it.
        HLG = k2.compose(H, LG, inner_labels="tokens")

        logging.info("Connecting LG")
        HLG = k2.connect(HLG)

        logging.info("Arc sorting LG")
        HLG = k2.arc_sort(HLG)
        logging.info(f"HLG.shape: {HLG.shape}")

        HLG_list.append(HLG)

    HLGs = k2.create_fsa_vec(HLG_list)
    return HLGs


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    if (lang_dir / "HLGs.pt").is_file():
        logging.info(f"{lang_dir}/HLG.pt already exists - skipping")
        return

    logging.info(f"Processing {lang_dir}")

    HLGs = compile_HLG(lang_dir)
    logging.info(f"Saving HLGs.pt to {lang_dir}")
    torch.save(HLGs.as_dict(), f"{lang_dir}/HLGs.pt")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
