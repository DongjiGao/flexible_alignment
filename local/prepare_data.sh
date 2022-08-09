#!/bin/bash

# Copyright 2021 Baidu USA (author Dongji Gao)

set -euo pipefail

stage=1

oov="<UNK>"

. ./path.sh
. utils/parse_options.sh

lang=$1
graph_dir=$2

if [ $stage -le 1 ]; then
  echo "$0: Stage 1, making lang dir (L.fst)"
  local/prepare_lang.sh \
    --sil-prob 0.0 \
    --position-dependent-phones false \
    data/local/dict \
    "${oov}" \
    data/local/tmp \
    ${lang}
fi

echo "Copy L_disambig.fst to ${graph_dir}"
cp ${lang}/L_disambig.fst.txt ${graph_dir}

echo "$0: Done."
exit 0
