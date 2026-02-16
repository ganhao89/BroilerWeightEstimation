"""
Video Preprocessing Script for Broiler Weight Estimation Pipeline.

Crops raw videos to a user-selected region and time range,
collects weight metadata, and saves results to test_videos/.

Usage:
    Single video:
        python video_preprocessing.py --input ../raw_videos/cam1.mp4

    Batch mode (same crop for all videos):
        python video_preprocessing.py --input_dir ../raw_videos/

    With time range:
        python video_preprocessing.py --input ../raw_videos/cam1.mp4 --start 00:00:05 --end 00:00:30
"""

import argparse
import csv
import os
import sys
import tkinter as tk
from pathlib import Path

import cv2

from VideoIOClass import VideoReader, VideoWriter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "test_videos"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"


def parse_time(time_str: str) -> float:
    """Parse a time string (HH:MM:SS or seconds) into seconds."""
    if time_str is None:
        return None
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def select_time_range_interactive(video_path: str) -> tuple:
    """
    Open interactive playback for the user to select start/end times.

    Controls:
        - Trackbar to seek through the video
        - 's' to mark start time
        - 'e' to mark end time
        - 'q' to confirm and close
        - Space to pause/resume playback
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    start_time = 0.0
    end_time = duration
    paused = False
    window_name = "Time Selection (s=start, e=end, Space=pause, q=confirm)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar(pos):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1), on_trackbar)

    print(f"Video duration: {format_time(duration)} ({total_frames} frames @ {fps:.1f} fps)")
    print("Controls: 's'=set start, 'e'=set end, Space=pause/play, 'q'=confirm")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps if fps > 0 else 0

        cv2.setTrackbarPos("Frame", window_name, min(current_frame, total_frames - 1))

        display = frame.copy() if not paused else frame.copy()
        info = f"Time: {format_time(current_time)} | Start: {format_time(start_time)} | End: {format_time(end_time)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if paused:
            cv2.putText(display, "PAUSED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord("s"):
            start_time = current_time
            print(f"Start time set: {format_time(start_time)}")
        elif key == ord("e"):
            end_time = current_time
            print(f"End time set: {format_time(end_time)}")
        elif key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if start_time > end_time:
        start_time, end_time = end_time, start_time
        print(f"Swapped start/end: {format_time(start_time)} -> {format_time(end_time)}")

    print(f"Selected range: {format_time(start_time)} - {format_time(end_time)}")
    return start_time, end_time


def select_crop_region(video_path: str, start_time: float) -> tuple:
    """
    Display the frame at start_time and let the user draw a bounding box.
    Returns (x, y, w, h) of the selected ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame at {format_time(start_time)}")

    print("Draw a bounding box on the frame. Press Enter/Space to confirm, 'c' to cancel.")
    roi = cv2.selectROI("Select Crop Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No crop region selected. Using full frame.")
        h, w = frame.shape[:2]
        return 0, 0, w, h

    print(f"Crop region: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    return roi


def ask_weight() -> float:
    """Show a tkinter dialog to input the broiler chicken weight in kg."""
    weight = [None]

    root = tk.Tk()
    root.title("Broiler Weight Input")
    root.geometry("300x150")
    root.resizable(False, False)

    tk.Label(root, text="Enter broiler chicken weight (kg):", font=("Arial", 11)).pack(pady=(15, 5))

    entry = tk.Entry(root, font=("Arial", 12), justify="center")
    entry.pack(pady=5)
    entry.focus_set()

    def on_ok(event=None):
        try:
            val = float(entry.get())
            weight[0] = val
            root.destroy()
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, "Invalid number")

    tk.Button(root, text="OK", command=on_ok, width=10).pack(pady=10)
    root.bind("<Return>", on_ok)

    root.mainloop()

    if weight[0] is None:
        print("No weight entered. Defaulting to 0.0 kg.")
        return 0.0

    print(f"Weight entered: {weight[0]} kg")
    return weight[0]


def save_metadata(video_name: str, weight_kg: float):
    """Append a row to the metadata CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = METADATA_CSV.exists()

    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_name", "weight_kg"])
        writer.writerow([video_name, weight_kg])

    print(f"Metadata saved: {video_name}, {weight_kg} kg")


def process_video(video_path: str, roi: tuple, start_time: float, end_time: float, weight_kg: float):
    """Crop the video spatially and temporally, then save to test_videos/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    output_path = str(OUTPUT_DIR / f"{video_name}.mp4")

    reader = VideoReader(video_path)
    fps, orig_w, orig_h = reader.get_info()

    x, y, w, h = roi
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total = end_frame - start_frame

    writer = VideoWriter(output_path, fps, (w, h))

    reader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = 0

    print(f"Processing: {video_name}")
    print(f"  Frames {start_frame} -> {end_frame} ({total} frames)")
    print(f"  Crop region: ({x}, {y}, {w}, {h})")

    while True:
        current = int(reader.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current >= end_frame:
            break

        ret, frame = reader.read()
        if not ret:
            break

        cropped = frame[y:y + h, x:x + w]
        writer.write(cropped)
        frame_count += 1

        if frame_count % 100 == 0 or frame_count == total:
            pct = frame_count / total * 100 if total > 0 else 100
            print(f"  Progress: {frame_count}/{total} ({pct:.1f}%)")

    reader.release()
    writer.release()

    print(f"  Saved: {output_path} ({frame_count} frames)")

    save_metadata(f"{video_name}.mp4", weight_kg)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw videos: crop spatially and temporally, record weight metadata."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a single input video")
    group.add_argument("--input_dir", type=str, help="Directory of input videos for batch mode")
    parser.add_argument("--start", type=str, default=None, help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", type=str, default=None, help="End time (HH:MM:SS or seconds)")

    args = parser.parse_args()

    # Collect video paths
    if args.input:
        video_paths = [args.input]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory")
            sys.exit(1)
        video_paths = sorted(
            str(p) for p in input_dir.iterdir()
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        )
        if not video_paths:
            print(f"No video files found in {args.input_dir}")
            sys.exit(1)

    print(f"Videos to process: {len(video_paths)}")
    for vp in video_paths:
        print(f"  {vp}")

    # Step 1: Time range selection
    first_video = video_paths[0]
    if args.start is not None and args.end is not None:
        start_time = parse_time(args.start)
        end_time = parse_time(args.end)
        print(f"Time range from CLI: {format_time(start_time)} - {format_time(end_time)}")
    else:
        start_time, end_time = select_time_range_interactive(first_video)

    # Step 2: Crop region selection (once, applied to all)
    roi = select_crop_region(first_video, start_time)

    # Step 3 & 4: Process each video
    for vp in video_paths:
        weight = ask_weight()
        process_video(vp, roi, start_time, end_time, weight)

    print("\nDone! All videos processed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")


if __name__ == "__main__":
    main()
