import os
import numpy as np
import pandas as pd
import trimesh
import argparse
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import imageio
import cv2

def run_reconstruction_pipeline(
    glb_path,
    csv_path,
    video_path,
    output_glb_folder="reconstructed_glbs",
    output_video_path="output_wireframe.mp4",
    scale_factor=3.534,
    sensor_width_mm=36,
    sensor_height_mm=24,
    focal_length_mm=25):

    os.makedirs(output_glb_folder, exist_ok=True)

    mesh = trimesh.load(glb_path, force='mesh')
    original_vertices = mesh.vertices.copy() * scale_factor
    faces = mesh.faces.copy()

    gltf = GLTF2().load(glb_path)
    camera_node_idx = next(i for i, node in enumerate(gltf.nodes) if node.camera is not None)
    camera_node = gltf.nodes[camera_node_idx]
    camera_position = np.array(camera_node.translation or [0, 0, 0])
    rotation_quat = camera_node.rotation or [0, 0, 0, 1]
    R_cam = R.from_quat(rotation_quat).as_matrix()

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fx = W * focal_length_mm / sensor_width_mm
    fy = H * focal_length_mm / sensor_height_mm
    cx = W / 2
    cy = H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def backproject_2d_to_3d(x, y, depth):
        x_norm = (x - K[0, 2]) / K[0, 0]
        y_norm = (y - K[1, 2]) / K[1, 1]
        ray_cam = np.array([x_norm, y_norm, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = R_cam.T @ ray_cam
        return camera_position + ray_world * depth

    def project_vertex(v):
        v_cam = R_cam @ (v - camera_position)
        if v_cam[2] <= 0:
            return None
        proj = K @ v_cam
        proj /= proj[2]
        return int(proj[0]), int(proj[1])

    df = pd.read_csv(csv_path)
    frame_count = df["frame_id"].max() + 1
    grouped = df.groupby("frame_id")

    vertex_depths = np.linalg.norm(original_vertices - camera_position, axis=1)

    video_reader = imageio.get_reader(video_path)
    video_writer = imageio.get_writer(output_video_path, fps=fps)

    for frame_idx, frame in tqdm(enumerate(video_reader), total=frame_count):
        if frame_idx >= frame_count:
            break

        updated_vertices = original_vertices.copy()

        if frame_idx in grouped.groups:
            tracked = grouped.get_group(frame_idx)
            for _, row in tracked.iterrows():
                pt_id = int(row["point_id"])
                if pt_id >= len(original_vertices):
                    continue
                x, y = row["x"], row["y"]
                depth = vertex_depths[pt_id]
                point_3d = backproject_2d_to_3d(x, y, depth)
                updated_vertices[pt_id] = point_3d

        mesh_deformed = trimesh.Trimesh(vertices=updated_vertices, faces=faces, process=False)
        glb_out_path = os.path.join(output_glb_folder, f"frame_{frame_idx:04d}.glb")
        mesh_deformed.export(glb_out_path)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for f in faces:
            i1, i2, i3 = f
            for i, j in [(i1, i2), (i2, i3), (i3, i1)]:
                p1 = project_vertex(updated_vertices[i])
                p2 = project_vertex(updated_vertices[j])
                if p1 is not None and p2 is not None:
                    cv2.line(frame_bgr, p1, p2, (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_writer.append_data(frame_rgb)

    video_writer.close()
    print(f"All .glb models are saved in: {output_glb_folder}")
    print(f"Video with visualization saved in: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Reconstruct and visualize 3D GLB models using tracked 2D points.")
    parser.add_argument("--glb_path", required=True, help="Path to the base GLB file with camera")
    parser.add_argument("--csv_path", required=True, help="Path to tracked_points.csv")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--output_glb_folder", default="reconstructed_glbs", help="Folder to save per-frame GLBs")
    parser.add_argument("--output_video_path", default="output_wireframe.mp4", help="Path to save output wireframe video")
    parser.add_argument("--scale_factor", type=float, default=3.534, help="Scaling factor for base mesh")
    parser.add_argument("--sensor_width_mm", type=float, default=36, help="Camera sensor width in mm")
    parser.add_argument("--sensor_height_mm", type=float, default=24, help="Camera sensor height in mm")
    parser.add_argument("--focal_length_mm", type=float, default=25, help="Camera focal length in mm")
    args = parser.parse_args()

    run_reconstruction_pipeline(
        glb_path=args.glb_path,
        csv_path=args.csv_path,
        video_path=args.video_path,
        output_glb_folder=args.output_glb_folder,
        output_video_path=args.output_video_path,
        scale_factor=args.scale_factor,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        focal_length_mm=args.focal_length_mm
    )

if __name__ == "__main__":
    main()
