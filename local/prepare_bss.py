#!/usr/bin/env python3

# 2022 Dongji Gao

import argparse
import os
import sys
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="This script do BSS on multi-channel audio.")
    parser.add_argument("wav_file", type=str, help="Input wav.scp file")
    parser.add_argument("tmp_dir", type=str, help="Directory to store temporary file")
    parser.add_argument("output_dir", type=str, help="Directory to BSSed audio")
    parser.add_argument("output_wav_file", type=str, help="Directory to BSSed audio")
    args = parser.parse_args()
    return args


def check_args(args):
    if not os.path.isfile(args.wav_file):
        print(f"wav file {args.wav_file} does not exists")
        sys.exit(1)

    if not os.path.isdir(args.tmp_dir):
        print(f"{args.tmp_dir} does not exist, building it")
        os.mkdir(args.tmp_dir)

    if not os.path.isdir(args.output_dir):
        print(f"{args.output_dir} does not exist, building it")
        os.mkdir(args.output_dir)


def prepocessing_audio(wav_file, tmp_dir):
    print("Preprocessing raw audio")
    session_dict = defaultdict(list)
    wav_location_dict = defaultdict(list)

    with open(wav_file, 'r') as wfile:
        for line in wfile.readlines():
            line_list = line.split()

            # pipeline form that need to be processed first
            # example:
            # "03-09-2019_LKC-Y2_T19_AE_L sox -G /export/corpora5/NTU/Collab_Audio/With_Annotations/03-09-2019_LKC-Y2_T19/AE/03-09-2019_LKC-Y2_T19_AE_U1850920A_L.wav -r 16000 -t wav -|"

            if len(line_list) > 2:
                filename = line_list[0].split(".")[0]
                session_id = "_".join(filename.split("_")[:-1])
                session_dict[session_id].append(filename)

                command = (" ".join(line_list[1:]))[:-1]
                command += f" > {tmp_dir}/{filename}.wav"

                print(f"Preprocessing file {filename}")
                os.system(command)

                wav_location_dict[filename] = f"{tmp_dir}/{filename}.wav"
            else:
                filename = line_list[0]
                session_id = "_".join(filename.split("_")[:-1])
                session_dict[session_id].append(filename)
                wav_location_dict[filename] = line_list[1]

    return session_dict, wav_location_dict


def combine_channels(session_dict, wav_location_dict, tmp_dir):
    for session in session_dict:
        channels_list = []
        for wav in session_dict[session]:
            location = wav_location_dict[wav]
            print(location)
            channels_list.append(location)
        channels = " ".join(channels_list)
        command = f"sox -M {channels} {tmp_dir}/{session}.wav"
        os.system(command)


def blind_source_separation(session_dict, tmp_dir, output_dir,
                            num_iter=15, segsize=15, fft_size=4096):
    print("Doing BSS")
    for session in session_dict:
        location = f"{tmp_dir}/{session}.wav"
        location = "mix4.wav"
        num_channels = len(session_dict[session])
        num_channels = 4

        command = "local/bss.py "
        command += f"{location} {output_dir}/{session}.wav "
        command += f"{num_iter} {segsize} {fft_size} {num_channels}"
        print(command)

        os.system(command)


def separate_channel(session_dict, output_dir):
    print("Separating channels of BSSed audio")
    for session in session_dict:
        location = f"{output_dir}/{session}.wav"

        c = 0
        for wav in enumerate(session_dict[session]):
            c += 1
            channel = 4
            command = f"sox {location} -b 16 "
            command += f"{output_dir}/{wav[1]}_{c}.wav remix {channel+1}"
            os.system(command)


def write_wav_scp(session_dict, output_dir, output_wav_file):
    with open(output_wav_file, 'w') as ws:
        for session in session_dict:
            for wav in session_dict[session]:
                location = f"{output_dir}/{wav}.wav"
                ws.write(f"{wav} {location}\n")


def main():
    args = get_args()
    check_args(args)

    wav_file = args.wav_file
    tmp_dir = args.tmp_dir
    output_dir = args.output_dir
    output_wav_file = args.output_wav_file

    session_dict, wav_location_dict = prepocessing_audio(wav_file, tmp_dir)
    combine_channels(session_dict, wav_location_dict, tmp_dir)
    blind_source_separation(session_dict, tmp_dir, output_dir)
    separate_channel(session_dict, output_dir)
    write_wav_scp(session_dict, output_dir, output_wav_file)

    print("Finish preprocessing original audio.")


if __name__ == "__main__":
    main()
