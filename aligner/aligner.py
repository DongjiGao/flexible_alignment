# 2021 Dongji Gao

import os
from pathlib import Path

import datasets
import k2
import numpy as np
import torch
import torch.nn.functional as F
from acceptor import Acceptor
from kaldialign import edit_distance
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_texts
from snowfall.decoding.graph import compile_HLG
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


class Aligner():
    def __init__(self, model, dataset, lang_dir, graph_dir):
        self.model = model
        self.dataset = datasets.load_from_disk(dataset)
        self.lang_dir = Path(lang_dir)
        self.graph_dir = Path(graph_dir)

    def make_g_fst(self):
        raise NotImplementedError

    def make_graph(self):
        raise NotImplementedError

    def load_model(self, model):
        print(f"loading model {model}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model)
        tokenizer = Wav2Vec2CTCTokenizer("./vocab.json",
                                         unk_token="<UNK>",
                                         pad_token="<eps>",
                                         word_dilimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16e3,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                           tokenizer=tokenizer)

    def load_graph(self, graph_dir):
        d = torch.load(graph_dir / "HLG.pt")
        self.HLG = k2.Fsa.from_dict(d)

    def decode(self):
        raise NotImplementedError

    def get_symbol_table(self, lang_dir):
        self.symbol_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    def run(self):
        raise NotImplementedError


class FlexibleAligner(Aligner):
    def __init__(self, model, dataset, lang_dir, graph_dir):
        super().__init__(model, dataset, lang_dir, graph_dir)
        pass

    def process_text(self, dataset):
        utt_2_spk = dict()
        spk_2_text = dict()
        spk_2_index = dict()
        id = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            utt_id = sample["utt_id"]
            spk_id = sample["spk_id"]

            text = sample["text"]
            if spk_id not in spk_2_text:
                spk_2_text[spk_id] = text
                spk_2_index[spk_id] = id
                id += 1
            utt_2_spk[utt_id] = spk_id
        return spk_2_text, utt_2_spk, spk_2_index

    def get_ses2spk(self, ses2spk_file):
        self.ses2spk_file = ses2spk_file

    def make_g_fst(self, spk_2_text, ses2spk, symbol_table, graph_dir, weight=0, skip_weight=0,
                   allow_deletion=True):
        self.acceptor = Acceptor(spk_2_text, ses2spk, graph_dir)
        self.acceptor.set_symbol_table(symbol_table)
        self.acceptor.set_boundary("@@")
        self.acceptor.run()

        print("Finish making G")

    def make_graph(self, lang_dir, graph_dir):
        G_list = list()
        HLG_list = list()

        phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
        word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')

        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(word_symbol_table)
        phone_ids = get_phone_symbols(phone_symbol_table)
        phone_ids_with_blank = [0] + phone_ids
        ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

        with open(graph_dir / "L_disambig.fst.txt") as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
            print("L loaded")
        with open(graph_dir / "G.fst.txt") as f:
            G_all = f.read().strip().split("\n\n")
            for G_single in G_all:
                G = k2.Fsa.from_openfst(G_single, acceptor=False)
                #                G_list.append(G)
                #            G = k2.create_fsa_vec(G_list)
                #            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                #        print("G loaded")

                HLG = compile_HLG(L=L,
                                  G=G,
                                  H=ctc_topo,
                                  labels_disambig_id_start=first_phone_disambig_id,
                                  aux_labels_disambig_id_start=first_word_disambig_id)
                HLG_list.append(HLG)

        HLGs = k2.create_fsa_vec(HLG_list)

        torch.save(HLGs.as_dict(), graph_dir / "HLG.pt")
        print("Finish making graph")

    def decode(self, dataset, HLGs, model, processor, symbol_table, graph_dir,
               utt_2_spk, spk_2_index,
               search_beam=30.0, output_beam=15.0,
               min_active_states=7000, max_active_states=28000, blank_bias=0.0):

        hyps_all = list()
        refs_all = list()

        for sample in dataset:

            utt_id = sample["utt_id"]
            print(f"decoding {utt_id}")
            spk_id = sample["spk_id"]
            HLG_index = torch.tensor([spk_2_index[spk_id]], dtype=torch.int32)
            HLG = k2.index_fsa(HLGs, HLG_index)

            if not hasattr(HLG, "lm_scores"):
                HLG.lm_scores = HLG.scores.clone()

            input_values = processor(sample["speech"],
                                     sampling_rate=16e3,
                                     return_tensors="pt", ).input_values
            with torch.no_grad():
                logits = model(input_values).logits
            nnet_output = F.log_softmax(logits,
                                        dim=-1,
                                        dtype=torch.float32)
            nnet_output[:, :, 0] += blank_bias
            supervision = torch.tensor([[0, 0, nnet_output.shape[1]]], dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
            lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, search_beam,
                                                 output_beam, min_active_states,
                                                 max_active_states)
            best_paths = k2.shortest_path(lattices, use_double_scores=True)
            indices = torch.tensor([0])
            hyps = get_texts(best_paths, indices)
            hyps_all.append([sample["utt_id"]] + [symbol_table.get(x) for x in hyps[0]])
            refs_all.append([sample["utt_id"]] + sample["text"].split())

        return hyps_all, refs_all

    def score(self, hyps_list, refs_list):
        dists = list()
        for index in range(len(refs_list)):
            dists.append(edit_distance(refs_list[index][1:], hyps_list[index][1:]))
        errors = {
            key: sum(dist[key] for dist in dists)
            for key in ["sub", "ins", "del", "total"]
        }
        total_words = sum(len(ref) for ref in refs_list)
        print("WER: {}({}/{})".format(errors["total"] / total_words,
                                      errors["total"], total_words))
        print("done scoring")
        print(errors)

    def run(self):
        dataset = self.dataset
        model = self.model
        graph_dir = self.graph_dir
        lang_dir = self.lang_dir

        self.load_model(model)
        spk_2_text, utt_2_spk, spk_2_index = self.process_text(dataset)
        self.get_symbol_table(lang_dir)
        self.make_g_fst(spk_2_text, self.ses2spk_file, self.symbol_table, graph_dir)
        self.make_graph(lang_dir, graph_dir)
        self.load_graph(graph_dir)

        HLG = self.HLG
        processor = self.processor
        hyps_list, refs_list = self.decode(dataset,
                                           HLG,
                                           self.model,
                                           processor,
                                           self.symbol_table,
                                           graph_dir,
                                           utt_2_spk,
                                           spk_2_index)
        hyp_text = []
        for hyp in hyps_list:
            print(hyp)
