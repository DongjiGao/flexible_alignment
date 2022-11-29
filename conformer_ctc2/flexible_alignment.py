#!/usr/bin/env python3
# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from lhotse import load_manifest
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
from icefall.env import get_env_info
import k2
from conformer import Conformer


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
            # parameters for conformer
            "subsampling_factor": 4,
            "feature_dim": 80,
            "nhead": 8,
            "dim_feedforward": 2048,
            "encoder_dim": 512,
            "num_encoder_layers": 12,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 14000,
            "max_active_states": 56000,
            "use_double_scores": False,
            "env_info": get_env_info(),
            # other
            "method": "1best",
            "num_decoder_layers": 0,
        }
    )
    return params

def build_dataloader(feats_dir):
    alignment_dls = []
    cuts = load_manifest(feats_dir / "cuts.jsonl.gz")

    if not isinstance(cuts, list):
        cuts = [cuts]
    for cut in cuts:
        k2_dataset = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            return_cuts=True,
        )
        sampler = SimpleCutSampler(
            cut,
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
)->Dict[str, List[List[str]]]:
    device = HLG.device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    print(feature.shape)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    nnet_output, memory, memory_key_padding_mask = model(feature)
    # nnet_output is (N, T, C)

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                params.subsampling_factor,
                rounding_mode="trunc",
            ), 
            torch.div(
                supervisions["num_frames"],
                params.subsampling_factor,
                rounding_mode="trunc",
            ),
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

    key = "no_rescore"
    hyps = get_texts(best_path)
    hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
    return {key: hyps}


def flexible_alignment(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLGs: k2.Fsa,
    lexicon: Lexicon,
):
    results = defaultdict(list)

    for batch_idx, batch in enumerate(dl):
        assert len(batch["supervisions"]["cut"]) == 1

        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        texts = batch["supervisions"]["text"]
        hlg_id = int(batch["supervisions"]["cut"][0].supervisions[0].hlg_id)
        HLG = HLGs[hlg_id]

        hyps_dict = align_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            batch=batch,
            lexicon=lexicon,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)

            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

        results[lm_scale].extend(this_batch)

    return results

def main():
    args = get_args()
    params = get_params()
    data_dir = Path(args.data_dir)
    lang_dir = Path(args.lang_dir)
    exp_dir = Path(args.exp_dir)

    setup_logger(f"{args.exp_dir}/log/flexible_alignment")
    logging.info(params)

    lexicon = Lexicon(lang_dir)
    max_phone_id = max(lexicon.tokens)
    num_classes = max_phone_id + 1

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    HLGs = k2.Fsa.from_dict(torch.load(f"{args.lang_dir}/HLGs.pt", map_location="cpu"))
    HLGs = HLGs.to(device)
    assert HLGs.requires_grad is False

    if not hasattr(HLGs, "lm_scores"):
        HLGs.lm_scores = HLGs.scores.clone()

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
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

    results = sorted(results_dict["no_rescore"], key=lambda x: int(x[0].split('_')[-1]))
    output_file = exp_dir / "text"
    with open(output_file, "w") as of:
        for result in results:
            utt_id, _, hyp_tokens_list = result
            hyp = " ".join(hyp_tokens_list)
            of.write(f"{utt_id} {hyp}\n")


if __name__ == "__main__":
    main()
