#!/usr/bin/env python3
# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

import argparse
import logging
from pathlib import Path
from functools import lru_cache

import torch
from torch import nn
from torch.utils.data import DataLoader
from lhotse import load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    one_best_decoding,
)
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon
import k2
from model import TdnnLstm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="data directory")
    parser.add_argument("--lang-dir", type=str, help="data directory")
    parser.add_argument("--checkpoint", type=str, help="checkpoint of model")
    parser.add_argument("--exp-dir", type=str, help="output directory")
    return parser.parse_args()


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 1024,
            "subsampling_factor": 3,
            "search_beam": 20,
            "output_beam": 5,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params

def build_dataloader(feats_dir):
    alignment_dls = []
    cuts = load_manifest_lazy(feats_dir / "cuts.jsonl.gz")

    if not isinstance(cuts, list):
        cuts = [cuts]
    for cut in cuts:
        k2_dataset = K2SpeechRecognitionDataset(
            input_stratepy=PrecomputedFeatures(),
            return_cuts=True,
        )
        sampler = SimpleCutSampler(
            cuts,
            max_cuts=1,
        )
        alignment_dl = DataLoader(
            k2_dataset, batch_size=None, sampler=sampler, num_workers=1
        )
        alignment_dls.append(alignment_dl)

    if isinstance(cut, list):
        return alignment_dls
    else:
        return alignment_dls[0]

def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    HLG: k2.Fsa,
    batch: dict,
    lexicon: Lexicon,
):
    device = HLG.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    feature = feature.permute(0, 2, 1)  # now feature is (N, C, T)
    print(feature.shape)

    nnet_output = model(feature)
    print(nnet_output.shape)
    # nnet_output is (N, T, C)

    supervisions = batch["supervisions"]
    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            supervisions["num_frames"] // params.subsampling_factor,
        ),
        1,
    ).to(torch.int32)

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )


def flexible_alignment(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLGs: k2.Fsa,
    lexicon: Lexicon,
):
    for batch_idx, batch in enumerate(dl):
        assert len(batch["supervisions"]["cut"] == 2
        hlg_id = batch["supervisions"]["cut"][0].supervisions[0].hlg_id
        HLG = HLGs[hlg_id]

        hyps_dict = align_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            batch=batch,
            lexicon=lexicon,
        )

def main():
    args = get_args()
    params = get_params()
    data_dir = Path(args.data_dir)
    lang_dir = Path(args.lang_dir)

    setup_logger(f"{args.exp_dir}/log/flexible_alignment")
    logging.info(params)

    lexicon = Lexicon(lang_dir)
    max_phone_id = max(lexicon.tokens)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    HLGs = k2.Fsa.from_dict(torch.load(f"{args.lang_dir}/HLGs.pt", map_location="cpu"))
    HLGs = HLGs.to(device)
    assert HLGs.requires_grad is False

    if not hasattr(HLGs, "lm_scores"):
        HLGs.lm_scores = HLGs.scores.clone()

    model = TdnnLstm(
        num_features=params.feature_dim,
        num_classes=max_phone_id + 1,  # +1 for the blank symbol
        subsampling_factor=params.subsampling_factor,
    )
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    model.eval()

    alignment_dls = build_dataloader(data_dir)

    results_dict = flexible_alignment(
        dl=alignment_dls,
        params=params,
        model=model,
        HLGs=HLGs,
        lexicon=lexicon,
    )


if __name__ == "__main__":
    main()
