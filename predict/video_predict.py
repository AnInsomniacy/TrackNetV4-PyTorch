#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tracknet_v4 import TrackNet


class TrackNetPredictor:
    def __init__(self, model_path, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNet().to(self.device)
        self.threshold = threshold

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (512, 288))
            frame = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            processed.append(frame)
        return torch.cat(processed, dim=0).unsqueeze(0).to(self.device)

    def predict(self, frames):
        input_tensor = self.preprocess_frames(frames)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0).cpu().numpy()

    def detect_ball(self, heatmap):
        if heatmap.max() < self.threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])


class SegmentedVideoProcessor:
    def __init__(self, model_path, dot_size=3, frames_per_segment=150, threshold=0.5):
        self.predictor = TrackNetPredictor(model_path, threshold)
        self.dot_size = dot_size
        self.frames_per_segment = self._adjust_segment_size(frames_per_segment)

    def _adjust_segment_size(self, frames):
        return frames - (frames % 3) if frames % 3 != 0 else frames

    def extract_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (width, height), fps, total_frames

    def extract_segment_frames(self, video_path, start_frame, num_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def group_frames(self, frames):
        groups = []
        for i in range(0, len(frames) - 2, 3):
            if i + 3 <= len(frames):
                groups.append(frames[i:i + 3])
        return groups

    def scale_coordinates(self, coords, original_size):
        if coords is None:
            return None
        x, y = coords
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 288
        return (int(x * scale_x), int(y * scale_y))

    def draw_ball(self, frame, ball_pos):
        if ball_pos is not None:
            cv2.circle(frame, ball_pos, self.dot_size, (0, 0, 255), -1)
        return frame

    def process_segment(self, frames, original_size, segment_idx):
        frame_groups = self.group_frames(frames)
        processed_frames = []
        ball_detected_count = 0

        with tqdm(total=len(frame_groups), desc=f"Segment {segment_idx + 1}", unit="groups") as pbar:
            for group in frame_groups:
                heatmaps = self.predictor.predict(group)

                for frame, heatmap in zip(group, heatmaps):
                    ball_pos_model = self.predictor.detect_ball(heatmap)
                    ball_pos_original = self.scale_coordinates(ball_pos_model, original_size)
                    processed_frame = self.draw_ball(frame.copy(), ball_pos_original)
                    processed_frames.append(processed_frame)

                    if ball_pos_original:
                        ball_detected_count += 1

                pbar.update(1)

        print(f"Ball detected in {ball_detected_count}/{len(processed_frames)} frames")

        return processed_frames, ball_detected_count

    def save_segment_video(self, frames, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not out.isOpened():
            raise RuntimeError(f"Cannot create video file: {output_path}")

        for frame in frames:
            out.write(frame)
        out.release()

    def merge_segments(self, segment_files, final_output, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_output, fourcc, fps, size)

        if not out.isOpened():
            raise RuntimeError(f"Cannot create final video file: {final_output}")

        with tqdm(total=len(segment_files), desc="Merging segments", unit="segments") as pbar:
            for segment_file in segment_files:
                cap = cv2.VideoCapture(segment_file)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
                pbar.update(1)

        out.release()
        print(f"Final video saved: {final_output}")

    def cleanup_segments(self, segment_files):
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                os.remove(segment_file)

    def process_video(self, video_path, output_path="processed_video.mp4"):
        print("Starting segmented shuttlecock detection...")
        print(f"Red dot size: {self.dot_size} pixels")
        print(f"Frames per segment: {self.frames_per_segment}")
        print(f"Detection threshold: {self.predictor.threshold}")

        original_size, fps, total_frames = self.extract_video_info(video_path)
        print(f"Video info: {total_frames} frames, {original_size[0]}x{original_size[1]}, {fps:.1f}FPS")

        num_segments = (total_frames + self.frames_per_segment - 1) // self.frames_per_segment
        print(f"Processing in {num_segments} segments")

        segment_files = []
        total_ball_detected = 0
        total_processed_frames = 0

        for seg_idx in range(num_segments):
            start_frame = seg_idx * self.frames_per_segment
            frames_to_extract = min(self.frames_per_segment, total_frames - start_frame)

            frames = self.extract_segment_frames(video_path, start_frame, frames_to_extract)
            processed_frames, ball_count = self.process_segment(frames, original_size, seg_idx)

            segment_output = f"temp_segment_{seg_idx:03d}.mp4"
            self.save_segment_video(processed_frames, segment_output, fps, original_size)
            segment_files.append(segment_output)

            total_ball_detected += ball_count
            total_processed_frames += len(processed_frames)

            del frames, processed_frames

        self.merge_segments(segment_files, output_path, fps, original_size)
        self.cleanup_segments(segment_files)

        detection_rate = (total_ball_detected / total_processed_frames) * 100
        print(f"Detection stats: {total_ball_detected}/{total_processed_frames} frames ({detection_rate:.1f}%)")

        return output_path


def main():
    model_path = "../best_v2.pth"
    input_video = "../predict_data/test.mp4"
    output_video = "../predict_data/processed_video.mp4"

    RED_DOT_SIZE = 7
    FRAMES_PER_SEGMENT = 150
    DETECTION_THRESHOLD = 0.2

    print("=" * 60)
    print("Badminton Shuttlecock Detection & Tracking System")
    print("=" * 60)
    print(f"Model file: {model_path}")
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Red dot size: {RED_DOT_SIZE} pixels")
    print(f"Frames per segment: {FRAMES_PER_SEGMENT}")
    print(f"Detection threshold: {DETECTION_THRESHOLD}")
    print("-" * 60)

    if not os.path.exists(input_video):
        print(f"Error: Input video file not found: {input_video}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    try:
        processor = SegmentedVideoProcessor(
            model_path,
            dot_size=RED_DOT_SIZE,
            frames_per_segment=FRAMES_PER_SEGMENT,
            threshold=DETECTION_THRESHOLD
        )
        output_path = processor.process_video(input_video, output_video)
        print("=" * 60)
        print(f"Processing complete! Output video: {output_path}")
        print("=" * 60)
    except Exception as e:
        print(f"Processing error: {str(e)}")


if __name__ == "__main__":
    main()
