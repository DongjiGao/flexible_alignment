# 2023 Dongji Gao

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
        self.disambig_token = "#0 #1 #2 #3 #4 #5 #6".split()

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
        if isinstance(symbol_table, str):
            try:
                self.symbol_table = SymbolTable.from_file(symbol_table)
            except ValueError:
                print(f"Can not get symbol table form file {words}")
        else:
            self.symbol_table = symbol_table

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
        speakers = set()
        try:
            with open(ses2spk_file, 'r') as s2s:
                for line in s2s.readlines():
                    session, speaker = line.split()
                    ses2spk[session].append(speaker)
                    spk2ses[speaker] = session
                    speakers.add(speaker)
        except ValueError:
            print(f"Can not open file {ses2spk_file}")

        spk2utt = defaultdict(list)
        utt2text = {}
        ses2order = defaultdict(list)
        spk2powerset = defaultdict(list)
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

                        for spk in speakers:
                            if spk == speaker:
                                if len(spk2powerset[spk]) == 0:
                                    spk2powerset[spk].append([utt])
                                else:
                                    spk2powerset[speaker][-1].append(utt)
                                    spk2powerset[speaker].append([utt])
                            else:
                                if len(spk2powerset[spk]) > 0:
                                    spk2powerset[spk][-1].append(utt)
        except ValueError:
            print(f"Can not open file {ses2spk_file}")

        return ses2spk, spk2utt, utt2text, ses2order, spk2powerset

    def build_single_baseline(self, utts, utt2text, epsilon_id, disambig_id, weight,
                              deletion_weight, output_handle):
        arcs = []
        existing_utts = set()

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        for utt in utts:
            cur_state = start_state
            text_id = utt2text[utt]
            assert len(text_id) >= 1

            text_id_str = str(text_id)
            if text_id_str not in existing_utts:
                existing_utts.add(text_id_str)

                for token_id in text_id[:-1]:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1

                token_id = text_id[-1]
                arc = self.get_arc(cur_state, final_state, token_id, token_id, weight)
                arcs.append(arc)

        final_arc = f"{final_state}\t{0}"
        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_single_2(self, utts, utt2text, epsilon_id, disambig_ids, weight,
                       deletion_weight, output_handle):
        arcs = []
        existing_utt = set()

        start_state = 0
        final_state = 1
        next_state = 2

        disambig_id = disambig_ids[0]
        auxiliary_arc = self.get_arc(start_state, next_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        cur_state = next_state
        next_state += 1

        for utt in utts:
            text_id = utt2text[utt]
            text_id_str = str(text_id)
            prev_skip_state = cur_state
            if text_id_str not in existing_utt:
                existing_utt.add(text_id_str)
                for token_id in text_id:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1
                arc = self.get_arc(prev_skip_state, cur_state, disambig_id, epsilon_id,
                                   deletion_weight)
                arcs.append(arc)

        auxiliary_arc = self.get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_single(self, utts, utt2text, epsilon_id, disambig_ids, weight,
                     deletion_weight, output_handle):
        arcs = []
        existing_utt = set()

        start_state = 0
        final_state = 1
        next_state = 2

        disambig_id = disambig_ids[0]
        auxiliary_arc = self.get_arc(start_state, next_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        cur_state = next_state
        next_state += 1

        for utt in utts:
            text_id = utt2text[utt]
            text_id_str = str(text_id)
            if text_id_str not in existing_utt:
                existing_utt.add(text_id_str)

                for token_id in text_id:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1
                arc = self.get_arc(start_state, cur_state, disambig_id, epsilon_id, weight)
                arcs.append(arc)
                arc = self.get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
                arcs.append(arc)

        auxiliary_arc = self.get_arc(cur_state, final_state, disambig_id, epsilon_id, weight)
        arcs.append(auxiliary_arc)
        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_multi_baseline(self, session, ses2spk, spk2utt, utt2text, ses2order, epsilon_id,
                             disambig_ids, weight, deletion_weight, output_handle,
                             allow_switch=True):
        arcs = []
        existing_utts = set()

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        for speaker in ses2spk[session]:
            utts = spk2utt[speaker]
            for utt in utts:

                cur_state = start_state
                text_id = utt2text[utt]
                assert len(text_id) >= 1

                text_id_str = str(text_id)
                if text_id_str not in existing_utts:
                    existing_utts.add(text_id_str)

                    for token_id in text_id[:-1]:
                        arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                        arcs.append(arc)
                        cur_state = next_state
                        next_state += 1

                    token_id = text_id[-1]
                    arc = self.get_arc(cur_state, final_state, token_id, token_id, weight)
                    arcs.append(arc)

        final_arc = f"{final_state}\t{0}"
        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_multi_powerset(self, session, ses2spk, spk2utt, utt2text, ses2order, epsilon_id,
                             disambig_ids, weight, deletion_weight, output_handle,
                             allow_switch=True):

        utt2state = {}
        arcs = []

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        # build for each single speaker
        dis_id = 0
        for speaker in ses2spk[session]:
            utts = spk2utt[speaker]
            dis_id += 1

            auxiliary_arc = self.get_arc(start_state, next_state, disambig_ids[dis_id], epsilon_id,
                                         weight)
            arcs.append(auxiliary_arc)
            cur_state = next_state
            next_state += 1

            for utt in utts[:-1]:
                prev_skip_state = cur_state
                utt_from_state = cur_state

                text_id = utt2text[utt]
                for token_id in text_id:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1

                utt_to_state = cur_state
                utt2state[utt] = (utt_from_state, utt_to_state)
                arc = self.get_arc(prev_skip_state, cur_state, disambig_ids[dis_id], epsilon_id,
                                   deletion_weight)
                arcs.append(arc)

                # build epsilon arc between utts to avoid epsilon cycle
                arc = self.get_arc(cur_state, next_state, disambig_ids[dis_id], epsilon_id, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1

            utt = utts[-1]
            text_id = utt2text[utt]
            prev_skip_state = cur_state
            utt_from_state = cur_state

            for token_id in text_id:
                arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1

            utt_to_state = cur_state
            utt2state[utt] = (utt_from_state, utt_to_state)
            arc = self.get_arc(prev_skip_state, cur_state, disambig_ids[dis_id], epsilon_id,
                               deletion_weight)
            arcs.append(arc)

            auxiliary_arc = self.get_arc(cur_state, final_state, disambig_ids[dis_id], epsilon_id,
                                         weight)
            arcs.append(auxiliary_arc)

        # build bypass between speakers
        if allow_switch:
            ordered_utts = ses2order[session]
            for index, utt in enumerate(ordered_utts[:-1]):
                next_utt = ordered_utts[index + 1]
                from_state = utt2state[utt][1]
                to_state = utt2state[next_utt][0]
                if from_state != to_state:
                    bypass_arc = self.get_arc(from_state, to_state, disambig_ids[0], epsilon_id,
                                              deletion_weight)
                    if bypass_arc not in arcs:
                        arcs.append(bypass_arc)

        final_arc = f"{final_state}\t{0}"
        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_multi_complex(self, session, ses2spk, spk2powerset, utt2text, epsilon_id,
                            disambig_ids, weight, deletion_weight, output_handle):

        arcs = []

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2
        cur_state = start_state

        dis_id = 0
        for speaker in ses2spk[session]:
            dis_id += 1
            for powerset in spk2powerset[speaker][:-1]:
                assert len(powerset) >= 3
                intra_utt = powerset[0]
                text_id = utt2text[intra_utt]
                for token_id in text_id:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1

                    arc = self.get_arc(state_state, cur_state, disambig_ids[0], disambig_ids[0],
                                       weight)
                    arcs.append(arc)
                    arc = self.get_arc(cur_state, final_state, disambig_ids[0], disambig_ids[0],
                                       weight)
                    arcs.append(arc)

                prev_final_state = cur_state
                for inter_utt in powerset[1:-2]:
                    prev_skip_state = cur_state

                    text_id = utt2text[inter_utt]
                    for token_id in text_id:
                        arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                        arcs.append(arc)
                        cur_state = next_state
                        next_state += 1
                    to_skip_state = cur_state
                    arc = self.get_arc(prev_skip_state, to_skip_state, disambig_ids[dis_id],
                                       epsilon_id, weight)
                    arcs.append(arc)

                text_id = powerset[-2]
                prev_skip_state = cur_state
                for token_id in text_id[:-1]:
                    arc = self.get_arc(cur_state, next_state, token_id, token_id, weight)
                    arcs.append(arc)
                    cur_state = next_state
                    next_state += 1
                token_id = text_id[-1]
                arc = self.get_arc(cur_state, prev_final_state, token_id, token_id, weight)
                arcs.append(arc)

                to_skip_state = prev_final_state
                arc = self.get_arc(prev_skip_state, to_skip_state, disambig_ids[dis_id],
                                   epsilon_id, weight)
                arcs.append(arc)

                cur_state = prev_final_state

    def build_multi_substring(self, session, ses2spk, spk2utt, utt2text, ses2order, epsilon_id,
                              disambig_ids, weight, deletion_weight, output_handle,
                              allow_switch=True):

        utt2state = {}
        arcs = []

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2

        # build for each single speaker
        dis_id = 0
        for speaker in ses2spk[session]:
            dis_id += 1
            utts = spk2utt[speaker]

            auxiliary_arc = self.get_arc(start_state, next_state, disambig_ids[dis_id], epsilon_id,
                                         weight)
            arcs.append(auxiliary_arc)
            cur_state = next_state
            next_state += 1

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

                arc = self.get_arc(start_state, cur_state, disambig_ids[dis_id], epsilon_id, weight)
                arcs.append(arc)
                arc = self.get_arc(cur_state, final_state, disambig_ids[dis_id], epsilon_id, weight)
                arcs.append(arc)

            auxiliary_arc = self.get_arc(cur_state, final_state, disambig_ids[dis_id], epsilon_id,
                                         weight)
            arcs.append(auxiliary_arc)

        # build bypass between speakers
        if allow_switch:
            ordered_utts = ses2order[session]
            for index, utt in enumerate(ordered_utts[:-1]):
                next_utt = ordered_utts[index + 1]
                from_state = utt2state[utt][1]
                to_state = utt2state[next_utt][0]
                if from_state != to_state:
                    bypass_arc = self.get_arc(from_state, to_state, disambig_ids[0], epsilon_id,
                                              deletion_weight)
                    if bypass_arc not in arcs:
                        arcs.append(bypass_arc)

        final_arc = f"{final_state}\t{0}"
        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build(self, ses2spk, spk2utt, utt2text, ses2order, spk2powerset, unk_id, epsilon_id,
              disambig_ids, output_dir, weight=0, deletion_weight=0):
        spk2accid = {}

        with open(output_dir / "G.fst.txt", 'w') as G:
            for index, session in enumerate(ses2spk):
                if len(ses2spk[session]) == 1:
                    speaker = ses2spk[session][0]
                    utts = spk2utt[speaker]
                    spk2accid[speaker] = index

                    self.build_single_baseline(utts, utt2text, epsilon_id, disambig_ids, weight,
                                               deletion_weight, G)
                else:
                    for speaker in ses2spk[session]:
                        spk2accid[speaker] = index

                    self.build_multi_powerset(session, ses2spk, spk2utt, utt2text, ses2order,
                                              epsilon_id, disambig_ids, weight, deletion_weight, G)

        return spk2accid

    def run(self):
        if self.symbol_table:
            symbol_table = self.symbol_table
        else:
            print("symbol_table not assigned")
            sys.exit(1)

        unk_id = symbol_table.get(self.unk_token)
        epsilon_id = symbol_table.get(self.epsilon_token)
        disambig_ids = []
        for token in self.disambig_token:
            disambig_ids.append(symbol_table.get(token))

        ses2spk, spk2utt, utt2text, ses2order, spk2powerset = self.process_file(self.text_file,
                                                                                self.ses2spk_file,
                                                                                symbol_table,
                                                                                unk_id)
        output_dir = self.output_dir
        spk2accid = self.build(ses2spk, spk2utt, utt2text, ses2order, spk2powerset, unk_id,
                               epsilon_id, disambig_ids, output_dir, deletion_weight=0)
        return spk2accid
