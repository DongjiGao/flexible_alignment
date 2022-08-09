#!/usr/bin/env python3

# 2020 Dongji Gao

import argparse
import torch
from collections import defaultdict
from pyannote.audio.utils.signal import Binarize


def get_args():
    parser = argparse.ArgumentParser(description="This script do VAD on input audio files and "
                                                 "analyze VAD performance.")
    parser.add_argument("--ref-segment", type=str, help="reference segments file")
    parser.add_argument("--collar", type=float, default=0.0, help="extend segments boundary")
    parser.add_argument("--gap", type=float, default=0.0, help="min nonsilence allowed")
    parser.add_argument("--metric", type=str, default="precision_recall", help="metric")
    parser.add_argument("wav_file", type=str, help="wav files")
    parser.add_argument("output_dir", type=str, help="output directory")

    args = parser.parse_args()
    return args


def get_ref_segments(ref_segment_file):
    ref_segments = defaultdict(list)
    with open(ref_segment_file, 'r') as rg:
        for line in rg.readlines():
            _, wav_id, start, end = line.split()
            start, end = float(start), float(end)
            ref_segments[wav_id].append((start, end))
    return ref_segments


def do_vad(wav_file, collar, gap):
    sad = torch.hub.load('pyannote/pyannote-audio', 'sad_dihard')

    segments_dict = defaultdict(list)
    with open(wav_file, 'r') as wf:
        for line in wf.readlines():
            print(line)
            line_list = line.split()
            assert len(line_list) == 2

            wav_id, location = line_list
            test_file = {'uri': wav_id, 'audio': location}

            sad_scores = sad(test_file)
            binarize = Binarize(offset=0.35, onset=0.35, log_scale=True,
                                min_duration_off=gap, min_duration_on=1.0)
            speech = binarize.apply(sad_scores, dimension=1)
            print(len(speech))
            for segment in speech:
                start = max(0, segment.start - collar)
                end = segment.end + collar
                segments_dict[wav_id].append((start, end))
        return segments_dict


def write_output(segments_dict, output_dir):
    segments = f"{output_dir}/segments"
    with open(segments, 'w') as seg:
        for wav_id in segments_dict:
            print(wav_id)
            for index, seg_tuple in enumerate(segments_dict[wav_id]):
                utt_id = f"{wav_id}_{index}"
                start_time = seg_tuple[0]
                end_time = seg_tuple[1]
                seg.write(f"{utt_id} {wav_id} {start_time:.2f} {end_time:.2f}\n")


def analyze(ref_segments_dict, hyp_segments_dict, metric):
    def get_overlap(ref_segment, hyp_segment):
        ref_start, ref_end = ref_segment
        hyp_start, hyp_end = hyp_segment

        max_start = max(hyp_start, ref_start)
        min_end = min(hyp_end, ref_end)
        overlap = max(0, min_end - max_start)
        return overlap

    def within_range(ref_segment, hyp_segment, range=1.0):
        ref_start, ref_end = ref_segment
        hyp_start, hyp_end = hyp_segment
        # if abs(ref_start - hyp_start) <= range and abs(ref_end - hyp_end) <= range:
        if abs(ref_end - hyp_end) <= range:
            return True
        return False

    def get_iou(ref_segment, hyp_segment):
        ref_start, ref_end = ref_segment
        hyp_start, hyp_end = hyp_segment

        intersection = max(0, min(ref_end, hyp_end) - max(ref_start, hyp_start))
        union = max(0, max(ref_end, hyp_end) - min(ref_start, hyp_start))
        iou = intersection / union
        return iou

    if metric == "precision_recall":
        overlap_all, ref_all, hyp_all = 0, 0, 0

        for session in hyp_segments_dict:
            assert session in ref_segments_dict
            total_overlap = 0
            for hyp_segment in hyp_segments_dict[session]:
                for ref_segment in ref_segments_dict[session]:
                    overlap = get_overlap(ref_segment, hyp_segment)
                    total_overlap += overlap
            ref_duration = sum(seg[1] - seg[0] for seg in ref_segments_dict[session])
            hyp_duration = sum(seg[1] - seg[0] for seg in hyp_segments_dict[session])

            overlap_all += total_overlap
            ref_all += ref_duration
            hyp_all += hyp_duration

            precision = total_overlap / hyp_duration
            recall = total_overlap / ref_duration
            print(f"For {session}, precision: {precision}, recall: {recall}")

        total_precision = overlap_all / hyp_all
        total_recall = overlap_all / ref_all
        print(f"In total, precision: {precision}, recall: {recall}")

    elif metric == "IoU":
        total_count = 0
        total_ious = []
        for session in hyp_segments_dict:
            count = 0
            ious = []
            assert session in ref_segments_dict
            for hyp_segment in hyp_segments_dict[session]:
                for ref_segment in ref_segments_dict[session]:
                    if within_range(ref_segment, hyp_segment):
                        count += 1
                        total_count += 1
                        iou = get_iou(ref_segment, hyp_segment)
                        ious.append(iou)
                        total_ious.append(iou)
            print(f"Within range rate is: {count / len(hyp_segments_dict[session])}")
            print(f"Average IoU is: {sum(ious) / len(ious)}")

        all = sum([len(hyp_segments_dict[x]) for x in hyp_segments_dict])
        ai = sum(total_ious) / len(total_ious)
        print(total_count/all, ai)

    elif metric == "jcard":
        for session in hyp_segments_dict:
            total_overlap = 0
            for hyp_segment in hyp_segments_dict[session]:
                for ref_segment in ref_segments_dict[session]:
                    overlap = get_overlap(ref_segment, hyp_segment)
                    total_overlap += overlap
            ref_duration = sum(seg[1] - seg[0] for seg in ref_segments_dict[session])
            hyp_duration = sum(seg[1] - seg[0] for seg in hyp_segments_dict[session])

            fa = hyp_duration - total_overlap
            miss = ref_duration - total_overlap
            total = hyp_duration + ref_duration - total_overlap
            jcard = (fa + miss) / total
            print(f"For {session}, Jaccard error rate is:{jcard}")


def main():
    args = get_args()
    hyp_segments_dict = do_vad(args.wav_file, args.collar, args.gap)
    write_output(hyp_segments_dict, args.output_dir)
    if args.ref_segment:
        ref_segments_dict = get_ref_segments(args.ref_segment)
        analyze(ref_segments_dict, hyp_segments_dict, args.metric)


if __name__ == "__main__":
    main()
