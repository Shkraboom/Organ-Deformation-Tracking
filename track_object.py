import os
import cv2
import torch
import argparse
import numpy as np
import imageio.v3 as iio
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def select_points(frame):
    selected = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", param)

    temp = frame.copy()
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    cv2.imshow("Select Points", temp)
    cv2.setMouseCallback("Select Points", click, temp)
    print("[INFO] Click on points to track, then press any key.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return selected

def run_tracking_pipeline(
    video_path,
    checkpoint_path="checkpoints/scaled_offline.pth",
    save_dir="./saved_videos",
    resize=1.0,
    target_fps=10,
    mask_path=None,
    grid_size=100,
    points_file=None
):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frame_step = max(int(original_fps // target_fps), 1)
    print(f"[INFO] Original FPS: {original_fps:.2f}, target FPS: {target_fps}, taking every {frame_step}th frame.")

    raw_frames = list(iio.imiter(video_path, plugin="FFMPEG"))
    sampled_frames = raw_frames[::frame_step]
    print(f"[INFO] Loaded {len(raw_frames)} frames → downsampled to {len(sampled_frames)} frames.")

    if resize != 1.0:
        frames = [cv2.resize(f, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_AREA) for f in sampled_frames]
        print(f"[INFO] Resized frames by factor {resize}.")
    else:
        frames = sampled_frames

    video_tensor = torch.tensor(np.stack(frames), dtype=torch.float32).permute(0, 3, 1, 2)[None].to(DEVICE)

    print("[INFO] Loading CoTracker model...")
    model = CoTrackerPredictor(checkpoint=checkpoint_path, offline=True).to(DEVICE)

    if mask_path:
        print("[INFO] Using segmentation mask for tracking...")
        segm_mask = np.array(Image.open(mask_path).convert("L"))
        segm_mask = torch.from_numpy(segm_mask).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size, segm_mask=segm_mask)
    else:
        if points_file:
            print("[INFO] Loading points from file...")
            ext = os.path.splitext(points_file)[1].lower()
            if ext == ".npy":
                selected_points = np.load(points_file)
            else:
                selected_points = np.loadtxt(points_file, dtype=np.int32)
        else:
            selected_points = select_points(frames[0])

        if not len(selected_points):
            raise ValueError("No points selected or loaded.")

        queries = torch.tensor([[0., float(x), float(y)] for (x, y) in selected_points], dtype=torch.float32)
        queries *= torch.tensor([1., resize, resize])  # scale x and y
        queries = queries[None].to(DEVICE)

        print("[INFO] Running prediction...")
        pred_tracks, pred_visibility = model(video_tensor, queries=queries)
        print(f"[INFO] Output shape: {pred_tracks.shape}")

        # === Сохраняем координаты ===
        print("[INFO] Saving point trajectories...")
        tracks_np = pred_tracks[0].cpu().numpy()
        num_frames, num_points = tracks_np.shape[:2]

        data = []
        for point_id in range(num_points):
            for frame_id in range(num_frames):
                x, y = tracks_np[frame_id, point_id]
                data.append({
                    "point_id": point_id,
                    "frame_id": frame_id,
                    "x": float(x),
                    "y": float(y)
                })

        os.makedirs(save_dir, exist_ok=True)
        coords_path = os.path.join(save_dir, "tracked_points.csv")
        pd.DataFrame(data).to_csv(coords_path, index=False)
        print(f"[INFO] Saved coordinates to {coords_path}")

    print("[INFO] Visualizing and saving results...")
    vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
    vis.visualize(video_tensor, pred_tracks, pred_visibility, query_frame=0)
    print(f"[INFO] Saved visualization to {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--checkpoint", default="checkpoints/scaled_offline.pth", help="Path to CoTracker checkpoint")
    parser.add_argument("--save_dir", default="./saved_videos", help="Directory to save results")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor for video frames")
    parser.add_argument("--target_fps", type=int, default=10, help="Downsample FPS to this value")
    parser.add_argument("--mask_path", type=str, help="Optional path to segmentation mask")
    parser.add_argument("--grid_size", type=int, default=100, help="Grid size for mask tracking")
    parser.add_argument("--points_file", type=str, help="Optional path to .npy or .txt with (N,2) points")
    args = parser.parse_args()

    run_tracking_pipeline(
        video_path=args.video_path,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        resize=args.resize,
        target_fps=args.target_fps,
        mask_path=args.mask_path,
        grid_size=args.grid_size,
        points_file=args.points_file
    )

if __name__ == "__main__":
    main()
