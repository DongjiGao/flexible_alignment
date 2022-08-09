# 2021 Dongji Gao

import os
import time
import json
from pathlib import Path

import datasets
import k2
import numpy as np
import torch
import torch.nn.functional as F
from acceptor import FlexibleAcceptor
from xvector import XVector
from kaldialign import edit_distance
from snowfall.common import find_first_disambig_symbol
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols
from icefall.decode import (
    Nbest,
    get_lattice,
    nbest_decoding,
)
from icefall.utils import get_texts
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

def log_softmax(x):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp

class Aligner:
    def __init__(self, model, dataset, lang_dir, graph_dir, output_dir):
        self.model = model
        self.dataset = datasets.load_from_disk(dataset)
        self.lang_dir = Path(lang_dir)
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    @staticmethod
    def make_g_fst(text, ses2spk, symbol_table, graph_dir):
        raise NotImplementedError

    @staticmethod
    def compile_HLG():
        raise NotImplementedError

    def make_graph(self, lang_dir, graph_dir):
        raise NotImplementedError

    @staticmethod
    def load_model(model_file, lang_dir):
        print(f"loading model {model_file}")
        model = Wav2Vec2ForCTC.from_pretrained(model_file)
        tokenizer = Wav2Vec2CTCTokenizer(lang_dir / "vocab.json",
                                         unk_token="<UNK>",
                                         pad_token="<eps>",
                                         word_dilimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16e3,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer)
        return model, processor

    @staticmethod
    def load_graph(graph_dir):
        d = torch.load(graph_dir / "HLG.pt")
        HLG = k2.Fsa.from_dict(d)
        return HLG

    def decode(self):
        raise NotImplementedError

    @staticmethod
    def get_symbol_table(lang_dir):
        symbol_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        return symbol_table

    def run(self):
        raise NotImplementedError


class FlexibleAligner(Aligner):
    def __init__(self, model, dataset, text, ses2spk, lang_dir, graph_dir, output_dir,
                 use_xvector=False, spk_similarity=None):
        super().__init__(model, dataset, lang_dir, graph_dir, output_dir)
        self.text = text
        self.ses2spk = ses2spk
        self.use_xvector = use_xvector
        if self.use_xvector:
            assert spk_similarity
            with open(spk_similarity, 'r') as file:
                self.similarity_dict = json.load(file)

    @staticmethod
    def make_g_fst(text, ses2spk, symbol_table, graph_dir):
        f_acceptor = FlexibleAcceptor(text, ses2spk, graph_dir)
        f_acceptor.set_symbol_table(symbol_table)
        spk2accid = f_acceptor.run()

        print("Finish making G")
        return spk2accid

    @staticmethod
    def compile_HLG_openfst(H, LG):
        LG = k2.arc_sort(LG)
        HLG = k2.compose(H, LG, inner_labels='phones')
        HLG = k2.connect(HLG)
        HLG = k2.arc_sort(HLG)

        return HLG

    @staticmethod
    def compile_HLG(H, L, G, first_token_disambig_id, first_word_disambig_id, determinize=False,
                    remove_epsilon=True):

        L = k2.arc_sort(L)
        #        G.labels[G.labels >= first_word_disambig_id] = 0
        #        G.__dict__["_properties"] = None
        G = k2.connect(G)
        G = k2.determinize(G)
        G = k2.arc_sort(G)
        G.lm_scores = G.scores.clone()
        LG = k2.compose(L, G)
        LG = k2.connect(LG)

        # determinize LG and remove disambig symbols
        if determinize:
            LG = k2.determinize(LG)
            LG = k2.connect(LG)

        LG.labels[LG.labels >= first_token_disambig_id] = 0
        # See https://github.com/k3-fsa/k2/issues/874
        # for why we need to set LG.properties to None
        LG.__dict__["_properties"] = None
        #        assert isinstance(LG.aux_labels, k2.RaggedTensor)
        #        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

        if remove_epsilon:
            LG = k2.remove_epsilon(LG)
            LG = k2.connect(LG)
            LG.aux_labels = LG.aux_labels.remove_values_eq(0)

        LG = k2.arc_sort(LG)
        HLG = k2.compose(H, LG, inner_labels='phones')
        HLG = k2.connect(HLG)
        HLG = k2.arc_sort(HLG)

        return HLG

    def make_graph(self, lang_dir, graph_dir, determinize=False, remove_epsilon=True,
                   openfst=False):
        HLG_list = []

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
            with open(graph_dir / "G_single.fst.txt", "w") as f:
                f.write(G_single)
            if openfst:
                command = f"/export/c26/dgao/flexible_alignment/ntu/local/make_lg.sh " \
                          f"{graph_dir}/G_single.fst.txt" \
                          f" {graph_dir}/L_disambig.fst.txt {lang_dir} {graph_dir}"
                os.system(command)

                with open(graph_dir / "LG.fst.txt") as f:
                    LG = k2.Fsa.from_openfst(f.read(), acceptor=False)
                    print("LG loaded from openfst")
                HLG = self.compile_HLG_openfst(H=ctc_topo, LG=LG)
            else:
                G = k2.Fsa.from_openfst(G_single, acceptor=False)
                HLG = self.compile_HLG(H=ctc_topo,
                                       L=L,
                                       G=G,
                                       first_token_disambig_id=first_phone_disambig_id,
                                       first_word_disambig_id=first_word_disambig_id,
                                       determinize=determinize,
                                       remove_epsilon=remove_epsilon,
                                       )
            HLG_list.append(HLG)

        HLGs = k2.create_fsa_vec(HLG_list)
        torch.save(HLGs.as_dict(), graph_dir / "HLG.pt")

        print(f"Finish compling HLG (determinize: {determinize}, remove_epsilon: {remove_epsilon})")

    def decode(self, dataset, HLGs, model, processor, symbol_table, graph_dir,
               spk2index, use_xvector, search_beam=30, output_beam=30,
               min_active_states=14000, max_active_states=56000, blank_bias=0.0):


        hyps_all = list()
        refs_all = list()
        prev_HLG_index = torch.tensor([-1], dtype=torch.int32)

        print(f"Start doing alignment")

        for sample in dataset:
            utt_id = sample["utt_id"]
            spk_id = sample["spk_id"]
            index_id = '_'.join(spk_id.split('_')[:-1])
            print("")
            print("===== nbest =====")
            print(f"Decoding utt: {utt_id}.")
            HLG_index = torch.tensor([spk2index[index_id]], dtype=torch.int32)

            if HLG_index != prev_HLG_index:
                HLG = k2.index_fsa(HLGs, HLG_index)
                assert HLG.requires_grad is False
                if not hasattr(HLG, "lm_scores"):
                    HLG.lm_scores = HLG.scores.clone()
                if HLG.device != self.device:
                    print(f"Moving HLG to {self.device}")
                    HLG = HLG.to(self.device)

                prev_HLG_index = HLG_index

            input_values = processor(sample["speech"],
                                     sampling_rate=16e3,
                                     return_tensors="pt",
                                     ).input_values
            input_values = input_values.to(self.device)

            with torch.no_grad():
                logits = model(input_values).logits
            nnet_output = F.log_softmax(logits,
                                        dim=-1,
                                        dtype=torch.float32)
            nnet_output[:, :, 0] += blank_bias
            supervision_segment = torch.tensor([[0, 0, nnet_output.shape[1]]], dtype=torch.int32)

            lattice = get_lattice(nnet_output=nnet_output,
                                  decoding_graph=HLG,
                                  supervision_segments=supervision_segment,
                                  search_beam=search_beam,
                                  output_beam=output_beam,
                                  min_active_states=min_active_states,
                                  max_active_states=max_active_states,
                                  )

            indices = torch.tensor([0])
            print(spk_id)
            print(sample['text'])

            nbest = Nbest.from_lattice(
                lattice=lattice,
                num_paths=5,
                use_double_scores=True,
                nbest_scale=1.0,
            )
            nbest = nbest.intersect(lattice)
            nbest_tot_scores = nbest.tot_scores().values

            if use_xvector:
                # compute cos distance
                spk_id_base = "_".join(spk_id.split('_')[:-2])
                num_hyps = nbest.shape.tot_size(1)

                print(f"Paths: {nbest.shape.tot_size(1)}")

                nbest_xvector_score = []
                print(f"n={num_hyps}")
                for i in range(num_hyps):
                    nbest_index = torch.tensor([i], dtype=torch.int32).to(self.device)
                    nbest_path = k2.index_fsa(nbest.fsa, nbest_index)

                    hyps = get_texts(nbest_path, indices)
                    hyps_token = [symbol_table.get(x) for x in hyps[0]]
                    print(hyps_token)

                    num_words = 0
                    mix_similarity = 0
                    for word in hyps_token:
                        if '_' in word:
                            suffix = word.split('_')[-1]
                            spk_id = spk_id_base + f"_{suffix}"

                            numerator = np.exp(self.similarity_dict[utt_id][spk_id])
                            denominator = 0
                            for d_spk_id in self.similarity_dict[utt_id]:
                                denominator += np.exp(self.similarity_dict[utt_id][d_spk_id])
                            mix_similarity += np.log(numerator) - np.log(denominator)
                            num_words += 1

                    if num_words > 0:
                        mix_similarity /= num_words
                    else:
                        mix_similarity = 0.0
                    nbest_xvector_score.append(mix_similarity)

                nbest_xvector_score = torch.tensor(nbest_xvector_score).to(self.device)
                nbest_xvector_score /= 10
                print(nbest_xvector_score)
                print(nbest_tot_scores / nbest_path.num_arcs)
                nbest_tot_scores = (nbest_tot_scores / nbest_path.num_arcs + nbest_xvector_score)

            print(nbest_tot_scores)
            index = nbest_tot_scores.argmax()
            index = index.to(torch.int32).unsqueeze(dim=0)
            best_path = k2.index_fsa(nbest.fsa, index)
            hyps = get_texts(best_path, indices)


            hyps_all.append([sample["utt_id"]] + [symbol_table.get(x) for x in hyps[0]])
            refs_all.append([sample["utt_id"]] + sample["text"].split())

        return hyps_all, refs_all

    @staticmethod
    def score(hyps_list, refs_list):
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
        print("Done scoring")
        print(errors)

    @staticmethod
    def write(hyp_list, output_dir):
        with open(output_dir / "text", "w") as output_file:
            for hyp in hyp_list:
                if len(hyp) > 1:
                    output_text = " ".join(hyp)
                    output_file.write(f"{output_text}\n")

    def run(self):
        dataset = self.dataset
        model_file = self.model
        graph_dir = self.graph_dir
        lang_dir = self.lang_dir
        text_file = self.text
        ses2spk_file = self.ses2spk
        output_dir = self.output_dir
        use_xvector = self.use_xvector

        # load model, graph
        model, processor = self.load_model(model_file, lang_dir)
        model.to(self.device)
        symbol_table = self.get_symbol_table(lang_dir)

        spk2index = self.make_g_fst(text_file, ses2spk_file, symbol_table, graph_dir)
        self.make_graph(lang_dir, graph_dir, determinize=True, remove_epsilon=True, openfst=True)
        HLGs = self.load_graph(graph_dir)

        # using GPU
        assert HLGs.requires_grad is False
        if not hasattr(HLGs, "lm_scores"):
            HLGs.lm_scores = HLGs.scores.clone()

        print(f"device is {self.device}")
        hyps_list, refs_list = self.decode(dataset,
                                           HLGs,
                                           model,
                                           processor,
                                           symbol_table,
                                           graph_dir,
                                           spk2index,
                                           use_xvector,
                                           )
        self.write(hyps_list, output_dir)
