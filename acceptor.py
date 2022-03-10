# 2022 Dongji Gao

from collections import defaultdict
from pathlib import Path

from k2 import SymbolTable


class Acceptor:
    def __init__(self, input_text, session2spk, output_dir):
        try:
            ses2spk = defaultdict(list)
            with open(session2spk, 'r') as session2spk:
                for line in session2spk.readlines():
                    line_list = line.split()
                    assert len(line_list) == 2

                    session, speaker = line_list
                    ses2spk[session].append(speaker)
            self.sesssion2spk = ses2spk
        except ValueError:
            print(f"Can not open file {session2spk}.")

        if isinstance(input_text, str):
            try:
                texts = {}
                with open(input_text, 'r') as tf:
                    for line in tf.readlines():
                        line_list = line.split()
                        if len(line_list) >= 2:
                            speaker = line_list[0]
                            texts[speaker] = line_list[1:]
                self.texts = texts
            except ValueError:
                print(f"Can not open file {input_text}.")
        else:
            self.texts = input_text

        self.unk_token = "<UNK>"
        self.epsilon_token = "<eps>"
        self.boundary_token = ""
        self.output_dir = Path(output_dir)

    def set_multi_speaker(self):
        pass

    def set_unk(self, unk_token):
        self.unk_token = unk_word

    def set_boundary(self, boundary_token):
        self.boundary_token = boundary_token

    def set_epsilon(self, eps_token):
        self.epsilon_token = eps_token

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    def get_symbol_table(self, words):
        try:
            self.symbol_table = SymbolTable.from_file(words)
        except ValueError:
            print(f"Can not get symbol table form file {words}")

    def text_to_symbol(self, text, unk_token, symbol_table):
        assert len(text) >= 1
        symbol_text = [symbol_table.get(word) if word in symbol_table else symbol_table.get(
            unk_token) for word in text]
        return symbol_text

    def get_arc(self, from_state, to_state, ilabel, olabel, weight):
        return f"{from_state}\t{to_state}\t{ilabel}\t{olabel}\t{weight}"

    def build_one(self, text_symbol, epsilon_symbol, unk_symbol, boundary_symbol,
                  weight, deletion_weight, output_handle):

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2
        cur_state = start_state

        arcs = []
        for symbol in text_symbol[:-1]:
            if symbol != boundary_symbol:
                arc = self.get_arc(cur_state, next_state, symbol, symbol, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1
            else:
                arc = self.get_arc(start_state, cur_state, boundary_symbol, epsilon_symbol,
                                   deletion_weight)
                arcs.append(arc)

                arc = self.get_arc(cur_state, final_state, boundary_symbol, epsilon_symbol,
                                   deletion_weight)
                arcs.append(arc)

        # last symbol
        symbol = text_symbol[-1]
        arc = self.get_arc(cur_state, final_state, symbol, symbol, weight)
        arcs.append(arc)

        # final state
        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build_one_multi(self, text_symbol_list, epsilon_symbol, unk_symbol, boundary_symbol,
                        weight, deletion_weight, output_handle):

        # start with the longest text
        max_id = -1
        max_boundary_counts = -1
        for id, text_symbol in enumerate(text_symbol_list):
            boundary_counts = text_symbol.count(boundary_symbol)
            if boundary_counts > max_boundary_counts:
                max_id = id
                max_boundary_counts = boundary_counts

        # Acceptor Topology
        start_state = 0
        final_state = 1
        next_state = 2
        cur_state = start_state

        arcs = []
        skip_states = []
        text_symbol = text_symbol_list[max_id]

        for symbol in text_symbol[:-1]:
            if symbol != boundary_symbol:
                arc = self.get_arc(cur_state, next_state, symbol, symbol, weight)
                arcs.append(arc)
                cur_state = next_state
                next_state += 1
            else:
                arc = self.get_arc(start_state, cur_state, boundary_symbol, epsilon_symbol,
                                   deletion_weight)
                arcs.append(arc)

                arc = self.get_arc(cur_state, final_state, boundary_symbol, epsilon_symbol,
                                   deletion_weight)
                arcs.append(arc)
                skip_states.append(cur_state)

        # last symbol
        symbol = text_symbol[-1]
        arc = self.get_arc(cur_state, final_state, symbol, symbol, weight)
        arcs.append(arc)

        # reuse the skip states
        # TODO: it's a bit complicated here, write more explanation
        cur_state = start_state
        for text_id, text_symbol in enumerate(text_symbol_list):
            # skip the longest text since we have already built it
            if text_id != max_id:
                skip_state_id = 0
                for symbol_id, symbol in enumerate(text_symbol[:-1]):
                    if symbol != boundary_symbol:
                        next_symbol = text_symbol[symbol_id + 1]
                        if next_symbol != boundary_symbol:
                            arc = self.get_arc(cur_state, next_state, symbol, symbol, weight)
                            arcs.append(arc)
                            cur_state = next_state
                            next_state += 1
                        else:
                            skip_state = skip_states[skip_state_id]
                            arc = self.get_arc(cur_state, skip_state, symbol, symbol,
                                               deletion_weight)
                            arcs.append(arc)
                            skip_state_id += 1
                    else:
                        cur_state = skip_state

                symbol = text_symbol[-1]
                arc = self.get_arc(cur_state, final_state, symbol, symbol, weight)
                arcs.append(arc)

        # final state
        final_arc = f"{final_state}\t{0}"

        for arc in arcs:
            output_handle.write(arc + "\n")
        output_handle.write(final_arc + "\n\n")

    def build(self, texts, ses2spk, epsilon_token, unk_token, symbol_table, output_dir,
              boundary_token,
              weight=0, deletion_weight=0, allow_deletion=True):

        epsilon_symbol = symbol_table.get(epsilon_token)
        unk_symbol = symbol_table.get(unk_token)

        if allow_deletion:
            assert boundary_token != ""
            try:
                boundary_symbol = symbol_table.get(boundary_token)
            except ValueError:
                print(f"Boundary '{boundary_token}' not in symbol table.")

        with open(output_dir / "G.fst.txt", 'w') as G:
            for session in ses2spk:
                if len(ses2spk[session]) == 1:
                    speaker = ses2spk[session][0]
                    text = texts[speaker]
                    text_symbol = self.text_to_symbol(text, unk_token, symbol_table)
                    self.build_one(text_symbol, epsilon_symbol, unk_symbol, boundary_symbol,
                                   weight, deletion_weight, G)
                else:
                    text_symbol_list = []
                    for speaker in ses2spk[session]:
                        text = texts[speaker]
                        text_symbol = self.text_to_symbol(text, unk_token, symbol_table)
                        text_symbol_list.append(text_symbol)
                    self.build_one_multi(text_symbol_list, epsilon_symbol, unk_symbol,
                                         boundary_symbol, weight, deletion_weight, G)

    def run(self):
        self.build(self.texts, self.sesssion2spk, self.epsilon_token, self.unk_token,
                   self.symbol_table, self.output_dir, self.boundary_token)
