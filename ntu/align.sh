#!/bin/bash

# 2022 Dongji Gao

# Flexible alignments


stage=0

# loading module

# script options

# acceptor options
oov="<UNK>"
dis="#0"
# [alignment, asr, ad]
task="alignment"


model="model/robust_seame_eng_ntu/checkpoint-54500"
dataset="dataset/ntu_bss_test"
ses2spk="../test/data/raw_toy/ses2spk"
text="../test/data/raw_toy/text_ref_switch"
lang="data/lang_ntu"
graph_dir="exp/graph" 
output_dir="data/ntu_collaborative_alignment"



. ./path.sh
. utils/parse_options.sh

for dir in ${output_dir} ${graph_dir} 
do
    [ ! -d ${dir} ] && mkdir -p ${dir}
done

export PYTHONPATH="/export/b14/dgao/snowfall":$PYTHONPATH
if [ $stage -le 0 ]; then
  echo "$0: Preparing data"
  local/prepare_data.sh \
    --oov ${oov} \
    ${lang} \
    ${graph_dir}
fi

if [ $stage -le 1 ]; then
  align_template.py ${model} ${dataset} ${text} ${ses2spk} ${lang} ${graph_dir} ${output_dir}
fi

echo "$0: Done"
