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
        "--allow-insertion",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--insertion-weight",
        type=float,
        default=0,
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

            session, speaker, utt_id = meta_id.split("_")

            if session not in ses2spk:
                ses2spk[session] = [speaker]
            else:
                if speaker not in ses2spk[session]:
                    ses2spk[session].append(speaker)

            spk2utt[speaker].append(utt_id)
            utt2text[utt_id] = text_list

        return ses2spk, spk2utt, utt2text


def make_single_substring(
    utts,
    utt2text,
    lexicon,
    unk_id,
    disambig_id,
    allow_insertion=False,
    insertion_weight=0,
):
    arcs = []
    text_ending_states = []
    start_state = 0
    next_state = 1
    cur_state = start_state

    for utt in utts:
        if allow_insertion:
            insertion_arc = get_arc(
                cur_state, cur_state, unk_id, unk_id, insertion_weight
            )
            arcs.append(insertion_arc)

        tokens = utt2text[utt]
        ids = tokens_to_ids(tokens, lexicon, unk_id)
        for id in ids:
            arc = get_arc(cur_state, next_state, id, id, 0)
            arcs.append(arc)
            cur_state = next_state
            next_state += 1
        skip_arc = get_arc(start_state, cur_state, disambig_id, 0, 0)
        arcs.append(skip_arc)
        text_ending_states.append(cur_state)

    prev_final_stage = cur_state
    final_state = next_state
    for state in text_ending_states[:-1]:
        skip_arc = get_arc(state, prev_final_stage, disambig_id, 0, 0)
        arcs.append(skip_arc)
    final_arc = get_arc(prev_final_stage, final_state, -1, -1, 0)
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

    eps_id = lexicon.word_table["<eps>"]
    assert eps_id == 0
    unk_id = lexicon.word_table[args.unk_token]
    disambig_id = lexicon.word_table[args.disambig_token]

    ses2spk, spk2utt, utt2text = preprocess(Path(args.text_file))

    with open(output_dir / "G.fst.txt", "w") as g_fst:
        with open(output_dir / "ses2hlg", "w") as s2h:
            for index, session in enumerate(ses2spk):
                if len(ses2spk[session]) == 1:
                    speaker = ses2spk[session][0]
                    utts = spk2utt[speaker]
                    fst_arcs = make_single_substring(
                        utts=utts,
                        utt2text=utt2text,
                        lexicon=lexicon,
                        unk_id=unk_id,
                        disambig_id=disambig_id,
                        allow_insertion=args.allow_insertion,
                        insertion_weight=args.insertion_weight,
                    )

                    for arc in fst_arcs:
                        g_fst.write(f"{arc}\n")
                    g_fst.write("\n")

                s2h.write(f"{session} {index}\n")


if __name__ == "__main__":
    main()
