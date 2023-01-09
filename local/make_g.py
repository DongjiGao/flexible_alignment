#!/usr/bin/env python3
# Copyright 2022 Johns Hopkins University (author: Dongji Gao)

import argparse
from collections import defaultdict
from pathlib import Path

from icefall.lexicon import Lexicon
from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=str, help="text file")
    parser.add_argument("--lang-dir", type=str, help="lang directory")
    parser.add_argument(
        "--level",
        choices=["word", "segment"],
        default="segment",
    )
    parser.add_argument(
        "--unk-token",
        type=str,
        default="<UNK>",
    )
    parser.add_argument(
        "--disambig-token",
        type=str,
        default="#0",
    )
    parser.add_argument(
        "--alignment-type",
        type=str,
        choices=["substring", "subsequence"],
        help="'substring' aligns consecutive segments, subseqnece' align any segments",
    )
    parser.add_argument(
        "--allow-insertion",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--insertion-weight",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--insertion-tokens",
        type=str,
        default="",
    )
    parser.add_argument("--output-dir", type=str, help="output dir")
    return parser.parse_args()


def get_arc(from_state, to_state, ilabel, olabel, weight):
    return f"{from_state} {to_state} {ilabel} {olabel} {weight}"


def tokens_to_ids(text_list, lexicon, unk_id):
    assert len(text_list) > 0
    ids = [
        lexicon.word_table[word] if word in lexicon.word_table else unk_id
        for word in text_list
    ]
    return ids


def preprocess(text_file):
    # text file:
    # session_spk_utt word1 word2 ...

    ses2spk = dict()
    spk2utt = defaultdict(list)
    utt2text = defaultdict(list)

    with open(text_file, "r") as tf:
        for line in tf.readlines():
            line_list = line.split()
            assert len(line_list) >= 2, f"Invalid text file: {text_file}"

            meta_id = line_list[0]
            text_list = line_list[1:]

            session, speaker, utt = meta_id.split("_")
            speaker_id = f"{session}_{speaker}"
            utt_id = meta_id

            if session not in ses2spk:
                ses2spk[session] = [speaker_id]
            else:
                if speaker_id not in ses2spk[session]:
                    ses2spk[session].append(speaker_id)

            spk2utt[speaker_id].append(utt_id)
            utt2text[utt_id] = text_list

        return ses2spk, spk2utt, utt2text


def make_single_subsequence(
    utts,
    utt2text,
    lexicon,
    unk_id,
    disambig_id,
    allow_insertion=False,
    insertion_weight=0,
    insertion_list=[],
):
    arcs = []
    start_state = 0
    next_state = 1
    cur_state = start_state
    utt_start_state = cur_state

    for utt_idx, utt in enumerate(utts):
        tokens = utt2text[utt]
        ids = tokens_to_ids(tokens, lexicon, unk_id)
        for id in ids:
            arc = get_arc(cur_state, next_state, id, id, 0)
            arcs.append(arc)
            cur_state = next_state
            next_state += 1
        utt_skip_arc = get_arc(utt_start_state, cur_state, disambig_id, 0, 0)
        arcs.append(utt_skip_arc)
        utt_start_state = cur_state

    prefinal_state = cur_state
    final_state = next_state

    if allow_insertion:
        for insertion_token_id in insertion_list:
            insertion_arc = get_arc(
                start_state,
                start_state,
                insertion_token_id,
                insertion_token_id,
                insertion_weight,
            )
            arcs.append(insertion_arc)

            insertion_arc = get_arc(
                prefinal_state,
                prefinal_state,
                insertion_token_id,
                insertion_token_id,
                insertion_weight,
            )
            arcs.append(insertion_arc)

    final_arc = get_arc(prefinal_state, final_state, -1, -1, 0)
    arcs.append(final_arc)

    # add final state
    arcs.append(f"{final_state}")
    arcs = sorted(arcs, key=lambda x: int(x.split()[0]))
    return arcs


def make_single_substring(
    utts,
    utt2text,
    lexicon,
    unk_id,
    disambig_id,
    level="segment",
    allow_insertion=False,
    insertion_weight=0,
    word_level=True,
    insertion_list=[],
):
    arcs = []
    boundary_states = []
    start_state = 0
    next_state = 1
    cur_state = start_state

    num_utts = len(utts)
    #    assert num_utts >= 2

    for utt_idx, utt in enumerate(utts):

        tokens = utt2text[utt]
        ids = tokens_to_ids(tokens, lexicon, unk_id)
        for id in ids:
            arc = get_arc(cur_state, next_state, id, id, 0)
            arcs.append(arc)
            cur_state = next_state
            next_state += 1
            if level == "word":
                boundary_states.append(cur_state)
        if level == "segment":
            boundary_states.append(cur_state)
        del boundary_states[-1]

    prefinal_state = cur_state
    final_state = next_state

    if allow_insertion:
        for insertion_token_id in insertion_list:
            insertion_arc = get_arc(
                start_state,
                start_state,
                insertion_token_id,
                insertion_token_id,
                insertion_weight,
            )
            arcs.append(insertion_arc)

            insertion_arc = get_arc(
                prefinal_state,
                prefinal_state,
                insertion_token_id,
                insertion_token_id,
                insertion_weight,
            )
            arcs.append(insertion_arc)

    for state in boundary_states:
        skip_arc = get_arc(start_state, state, disambig_id, 0, 0)
        arcs.append(skip_arc)
        skip_arc = get_arc(state, prefinal_state, disambig_id, 0, 0)
        arcs.append(skip_arc)
    final_arc = get_arc(prefinal_state, final_state, -1, -1, 0)
    arcs.append(final_arc)

    # add final state
    arcs.append(f"{final_state}")
    # k2 quires arcs of FSA ordered by from_state
    arcs = sorted(arcs, key=lambda x: int(x.split()[0]))
    return arcs


def main():
    args = get_args()
    lexicon = Lexicon(Path(args.lang_dir))
    output_dir = Path(args.output_dir)
    alignment_type = args.alignment_type

    unk_id = lexicon.word_table[args.unk_token]
    if args.insertion_tokens:
        insertion_tokens_path = Path(args.insertion_tokens)
        if not insertion_tokens_path.is_file():
            raise ValueError(
                f"Insertion token file {insertion_tokens_path} does not exist."
            )
        insertion_set = set()
        with insertion_tokens_path.open("r") as f:
            for line in f.readlines():
                token = line.split()[0]
                token_id = lexicon.word_table[token]
                insertion_set.add(token_id)
        insertion_list = list(insertion_set)
    else:
        insertion_list = [unk_id]

    eps_id = lexicon.word_table["<eps>"]
    assert eps_id == 0
    disambig_id = lexicon.word_table[args.disambig_token]

    ses2spk, spk2utt, utt2text = preprocess(Path(args.text_file))

    with open(output_dir / "G.fst.txt", "w") as g_fst:
        with open(output_dir / "ses2hlg", "w") as s2h:
            for index, session in enumerate(ses2spk):
                if len(ses2spk[session]) == 1:
                    speaker = ses2spk[session][0]
                    utts = spk2utt[speaker]
                    if alignment_type == "substring":
                        fst_arcs = make_single_substring(
                            utts=utts,
                            utt2text=utt2text,
                            lexicon=lexicon,
                            unk_id=unk_id,
                            disambig_id=disambig_id,
                            level=args.level,
                            allow_insertion=args.allow_insertion,
                            insertion_weight=args.insertion_weight,
                            insertion_list=insertion_list,
                        )
                    elif alignment_type == "subsequence":
                        fst_arcs = make_single_subsequence(
                            utts=utts,
                            utt2text=utt2text,
                            lexicon=lexicon,
                            unk_id=unk_id,
                            disambig_id=disambig_id,
                            allow_insertion=args.allow_insertion,
                            insertion_weight=args.insertion_weight,
                            insertion_list=insertion_list,
                        )

                    for arc in fst_arcs:
                        g_fst.write(f"{arc}\n")
                    g_fst.write("\n")

                s2h.write(f"{session} {index}\n")


if __name__ == "__main__":
    main()
