#!/usr/bin/env python3

# 2022 Dongji Gao

from pathlib import Path

from k2 import SymbolTable


class Acceptor:
    def __init__(self, text_file, spk2session, output_dir):
        try:
            s2s = {}
            with open(spk2session, 'r') as spk2session:
                for line in spk2session.readlines():
                    line_list = line.split()
                    assert len(line_list) == 2

                    speaker, session = line_list
                    assert speaker not in s2s
                    s2s[speaker] = session
            self.spk2session = s2s
        except ValueError:
            print(f"Can not open file {spk2session}.")

        try:
            texts = {}
            with open(text_file, 'r') as tf:
                for line in tf.readlines():
                    line_list = line.split()
                    if len(line_list) >= 2:
                        speaker = line_list[0]
                        texts[speaker] = line_list[1:]
            self.texts = texts
        except ValueError:
            print(f"Can not open file {spk2session}.")

        self.unk_token = "<UNK>"
        self.epsilon_token = "<eps>"
        self.boundary_token = ""
        self.output_dir = Path(output_dir)
        self.multi_speaker = False

    def set_multi_speaker(self):
        self.multi_speaker = True

    def set_unk(self, unk_token):
        self.unk_token = unk_word

    def set_boundary(self, boundary_token):
        self.boundary_token = boundary_token

    def set_epsilon(self, eps_token):
        self.epsilon_token = eps_token

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

    def build_single(self, text_symbol, epsilon_symbol, unk_symbol, boundary_symbol, weight,
                     deletion_weight, output_handle):

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

    def build(self, texts, epsilon_token, unk_token, symbol_table, output_dir, boundary_token,
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
            for speaker in texts:
                text = texts[speaker]
                text_symbol = self.text_to_symbol(text, unk_token, symbol_table)
                self.build_single(text_symbol, epsilon_symbol, unk_symbol, boundary_symbol, weight,
                                  deletion_weight, G)

    def build_multi_speaker(self, texts, epsilon_token, symbol_table, output_dir, weight=0,
                            deletion_weight=0, boundary_token="@@", allow_deletion=True):
        pass

    def run(self):
        self.build(self.texts, self.epsilon_token, self.unk_token, self.symbol_table,
                   self.output_dir, self.boundary_token)
