#!/usr/bin/env python3

# 2022 Dongji Gao

import sys
from collections import defaultdict
import numpy as np
import kaldi_io


class XVector:
#    def __init__(self):
#        self.ses2spk = ses2spk
#        self.spk2utt = spk2utt
#
##        with open(ses2spk, 'r') as ses2spk:
#            for line in ses2spk.readlines():
#                sesssion, speaker = line.split()
#                self.ses2spk[sesssion].append(speaker)
#
#        with open(spk2utt, 'r') as spk2utt:
#            for line in spk2utt.readlines():
#                spk, utt = line.split()
#                self.spk2utt[spk].append(utt)

    def read_xvector(self, scp_file):
        embedding_dict = {}
        for utt_id, vec in kaldi_io.read_vec_flt_scp(scp_file):
            vec = np.array(vec)
            embedding_dict[utt_id] = vec
        return embedding_dict

    def compute_cos_distance(self, a, b):
        return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
