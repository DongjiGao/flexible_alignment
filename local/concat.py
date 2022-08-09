#!/usr/bin/env python3

import sys
text = sys.argv[1]
output_text = sys.argv[2]
id = sys.argv[3]

def get_text_dict(text):
    text_dict = {}
    with open(text, 'r') as txt:
        for line in txt.readlines():
            line_list = line.split()
            if id == "utt_id":
                spk_id = "_".join(line_list[0].split("_")[:-2])
            elif id == "spk_id":
                spk_id = line_list[0]
            text = line_list[1:]

            if spk_id not in text_dict:
                text_dict[spk_id] = text
            else:
                text_dict[spk_id] += text
    return text_dict

text_dict = get_text_dict(text)

with open(output_text, 'w') as ot:
    for spk_id in text_dict:
        text = " ".join(text_dict[spk_id])
        ot.write(f"{spk_id} {text}\n")
