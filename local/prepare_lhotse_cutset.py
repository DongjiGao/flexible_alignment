#!/usr/bin/env python3
# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

import argparse
from pathlib import Path

from lhotse import (
    CutSet, 
    S3PRLSSL, 
    S3PRLSSLConfig,
    NumpyFilesWriter,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter
)

from lhotse.kaldi import load_kaldi_data_dir

from icefall.utils import get_executor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="data directory")
    parser.add_argument("--lang-dir", type=str, help="lang directory")
    parser.add_argument(
        "--feature-type",
        choices=["ssl", "fbank"],
        help="acoustic feature type"
    )
    return parser.parse_args()


def get_ses2hlg(lang_dir):
    ses2hlg = {}
    with open(lang_dir / "ses2hlg", "r") as s2h:
        for line in s2h.readlines():
            session_id, hlg_id = line.split()
            assert session_id not in ses2hlg
            ses2hlg[session_id] = hlg_id
    return ses2hlg


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    lang_dir = Path(args.lang_dir)
    feature_type = args.feature_type
    ses2hlg = get_ses2hlg(lang_dir)

    recording_set, supervision_set, _ = load_kaldi_data_dir(
        data_dir, sampling_rate=16000
    )
    for supervision in supervision_set:
        meta_id = supervision.id
        session, _, _ = meta_id.split("_")
        assert session in ses2hlg
        supervision.hlg_id = ses2hlg[session]


    if feature_type == "ssl":
        extractor = S3PRLSSL(
            S3PRLSSLConfig(ssl_model="wav2vec2_large_ll60k", device="cuda")
        )
        storage_type = NumpyFilesWriter
        num_jobs = 1
    elif feature_type == "fbank":
        num_mel_bins = 80
        extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
        storage_type = LilcomChunkyWriter
        num_jobs = 15
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    # build cutset
    with get_executor() as ex:
        cuts_filename = f"cuts.jsonl.gz"
        cut_set = CutSet.from_manifests(
            recordings=recording_set,
            supervisions=supervision_set,
        )
        # get small cuts from supervision (segments)
        cut_set = cut_set.trim_to_supervisions().to_eager()
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{data_dir}/{feature_type}_feats",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=storage_type,
        )
        cut_set.to_file(data_dir / cuts_filename)


if __name__ == "__main__":
    main()
