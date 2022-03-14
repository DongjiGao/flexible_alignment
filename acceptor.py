# 2022 Dongji Gao

import sys
from collections import defaultdict
from pathlib import Path

from k2 import SymbolTable


# the base Acceptor
class Acceptor:
    def __init__(self, text_file, ses2spk_file):
        self.text_file = text_file
        self.ses2spk_file = ses2spk_file

        self.symbol_table = ''
        self.unk_token = "<UNK>"
        self.epsilon_token = "<eps>"
        self.disambig_token = "#0"

    @staticmethod
    def process_file(text_file, ses2spk_file, symbol_table, unk_id):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def set_unk(self, unk_token):
        self.unk_token = unk_token

    def set_epsilon(self, eps_token):
        self.epsilon_token = eps_token

    def set_disambig(self, disambig_token):
        self.disambig_token = disambig_token

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    def get_symbol_table(self, words):
        try:
            self.symbol_table = SymbolTable.from_file(words)
        except ValueError:
            print(f"Can not get symbol table form file {words}")

    @staticmethod
    def token_to_id(text, symbol_table, unk_id):
        assert len(text) > 0
        text_id = [symbol_table.get(word) if word in symbol_table else unk_id for word in text]
        return text_id

    @staticmethod
    def get_arc(from_state, to_state, ilabel, olabel, weight):
        return f"{from_state}\t{to_state}\t{ilabel}\t{olabel}\t{weight}"


class FlexibleAcceptor(Acceptor):
    def __init__(self, text_file, ses2spk_file, output_dir):
        super().__init__(text_file, ses2spk_file)
        self.output_dir = Path(output_dir)

    @staticmethod
    def process_file(text_file, ses2spk_file, symbol_table, unk_id):
        ses2spk = defaultdict(list)
        spk2ses = {}
        try:
            with open(ses2spk_file, 'r') as s2s:
                for line in s2s.readlines():
                    session, speaker = line.split()
                    ses2spk[session].append(speaker)
                    spk2ses[speaker] = session
        except ValueError:
            print(f"Can not open file {ses2spk_file}")

        spk2utt = defaultdict(list)
        utt2text = {}
        ses2order = defaultdict(list)
        spk_id = defaultdict(int)
        try:
            with open(text_file, 'r') as tf:
                for line in tf.readlines():
                    line_list = line.split()
                    if len(line_list) >= 2:
                        speaker = line_list[0]
                        text = line_list[1:]
                        text_id = Acceptor.token_to_id(text, symbol_table, unk_id)

                        id = spk_id[speaker]
                        spk_id[speaker] += 1
                        utt = f"{speaker}_{id}"

                        spk2utt[speaker].append(utt)
                        utt2text[utt] = text_id

                        session = spk2ses[speaker]
                        ses2order[session].append(utt)
        except ValueError:
            print(f"Can not open file {ses2spk_file}")

        return ses2spk, spk2utt, utt2text, ses2order


    def build_single(self, utts, utt2text, epsilon_id, disambig_id, weight,
                     deletion_weight, output_handle):
        # final_id is for arcs to the final state in k2 FST
        utt2state = {}
        arcs = []

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        auxiliary_arc = self.get_arc(start_state, next_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        cur_state = next_state
        next_state += 1
        prev_skip_state = cur_state

        for utt in utts:
            text_id = utt2text[utt]
            utt_from_state = cur_state

            for token_id in text_id:
                arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1
            utt_to_state = cur_state
            utt2state[utt] = (utt_from_state, utt_to_state)

            arc = self.get_arc(prev_skip_state, cur_state, disambig_id, epsilon_id, deletion_weight)
            arcs.append(arc)
            prev_skip_state = cur_state

        auxiliary_arc = self.get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_multi(self, session, ses2spk, spk2utt, utt2text, ses2order,
                    epsilon_id, disambig_id, weight, deletion_weight, output_handle):

        utt2state = {}
        arcs = []

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        # build for each single speaker
        for speaker in ses2spk[session]:
            utts = spk2utt[speaker]

            auxiliary_arc = self.get_arc(start_state, next_state, disambig_id, epsilon_id, weight)
            arcs.append(auxiliary_arc)
            cur_state = next_state
            next_state += 1
            prev_skip_state = cur_state

            for utt in utts:
                text_id = utt2text[utt]
                utt_from_state = cur_state

                for token_id in text_id:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1
                utt_to_state = cur_state
                utt2state[utt] = (utt_from_state, utt_to_state)

                arc = self.get_arc(prev_skip_state, cur_state, disambig_id, epsilon_id,
                                   deletion_weight)
                arcs.append(arc)
                prev_skip_state = cur_state

            auxiliary_arc = self.get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
            arcs.append(auxiliary_arc)

        # build bypass between speakers
        ordered_utts = ses2order[session]
        for index, utt in enumerate(ordered_utts[:-1]):
            next_utt = ordered_utts[index + 1]
            from_state = utt2state[utt][1]
            to_state = utt2state[next_utt][0]
            if from_state != to_state:
                bypasa_arc = self.get_arc(from_state, to_state, 11, epsilon_id,
                                          deletion_weight)
                arcs.append(bypasa_arc)

        final_arc = f"{final_state}\t{0}"
        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build(self, ses2spk, spk2utt, utt2text, ses2order, unk_id, epsilon_id, disambig_id,
              output_dir, weight=0, deletion_weight=0):
        spk2accid = {}
        with open(output_dir / "G.fst.txt", 'w') as G:
            for index, session in enumerate(ses2spk):
                if len(ses2spk[session]) == 1:
                    speaker = ses2spk[session][0]
                    utts = spk2utt[speaker]
                    spk2accid[speaker] = index

                    self.build_single(utts, utt2text, epsilon_id, disambig_id, weight,
                                      deletion_weight, G)
                else:
                    text_symbol_list = []
                    for speaker in ses2spk[session]:
                        spk2accid[speaker] = index

                    self.build_multi(session, ses2spk, spk2utt, utt2text, ses2order,
                                     epsilon_id, disambig_id, weight, deletion_weight, G)

        return spk2accid

    def run(self):
        if self.symbol_table:
            symbol_table = self.symbol_table
        else:
            print("symbol_table not assigned")
            sys.exit(1)

        unk_id = symbol_table.get(self.unk_token)
        epsilon_id = symbol_table.get(self.epsilon_token)
        disambig_id = symbol_table.get(self.disambig_token)

        ses2spk, spk2utt, utt2text, ses2order = self.process_file(self.text_file,
                                                                  self.ses2spk_file,
                                                                  symbol_table,
                                                                  unk_id)
        output_dir = self.output_dir
        spk2accid = self.build(ses2spk, spk2utt, utt2text, ses2order, unk_id, epsilon_id,
                               disambig_id, output_dir)
        return spk2accid
