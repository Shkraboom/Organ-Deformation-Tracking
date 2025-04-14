import os
import cv2
import argparse
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R
from math import atan

os.environ["PYOPENGL_PLATFORM"] = "egl"

def render_3d_overlay_on_video(
    glb_folder,
    glb_path_with_camera,
    video_path,
    output_video_path="output_3d_on_video.mp4",
    sensor_width_mm=36,
    sensor_height_mm=24,
    focal_length_mm=25):
 
    gltf = GLTF2().load(glb_path_with_camera)
    camera_node_idx = next(i for i, node in enumerate(gltf.nodes) if node.camera is not None)
    camera_node = gltf.nodes[camera_node_idx]
    translation = camera_node.translation or [0, 0, 0]
    rotation_quat = camera_node.rotation or [0, 0, 0, 1]

    camera_position = np.array(translation)
    rotation_matrix = R.from_quat(rotation_quat).as_matrix()

    aspect_ratio = sensor_width_mm / sensor_height_mm
    yfov = 2 * atan((sensor_height_mm / 2) / focal_length_mm)

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = camera_position

    video_cap = cv2.VideoCapture(video_path)
    assert video_cap.isOpened(), "Failed to open video"
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    image_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (image_width, image_height))

    glb_files = sorted([
        os.path.join(glb_folder, f)
        for f in os.listdir(glb_folder)
        if f.endswith(".glb")
    ])

    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    for idx in tqdm(range(min(len(glb_files), frame_count)), desc="Rendering with background video"):
        ret, frame_bgr = video_cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mesh_trimesh = trimesh.load(glb_files[idx], force='mesh')
        mesh_trimesh.visual.face_colors = [150, 150, 150, 255]  #gray

        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.2, 0.2, 0.2])
        scene.add(mesh_pyrender)
        scene.add(camera, pose=camera_pose)

        light1 = pyrender.PointLight(color=np.ones(3), intensity=10.0)
        light_pose1 = np.eye(4)
        light_pose1[:3, 3] = camera_position + rotation_matrix @ np.array([0, 0, 2])
        scene.add(light1, pose=light_pose1)

        light2 = pyrender.PointLight(color=np.ones(3), intensity=7.0)
        light_pose2 = np.eye(4)
        light_pose2[:3, 3] = camera_position + rotation_matrix @ np.array([2, 2, 2])
        scene.add(light2, pose=light_pose2)

        color_rgb, _ = renderer.render(scene)

        mask = np.any(color_rgb != [255, 255, 255], axis=-1)
        blended_rgb = frame_rgb.copy()
        blended_rgb[mask] = color_rgb[mask]

        blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)
        writer.write(blended_bgr)

    video_cap.release()
    writer.release()
    renderer.delete()
    print(f"Video with 3D model overlay saved: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Render 3D GLB overlays on top of a video")
    parser.add_argument("--glb_folder", required=True, help="Folder with per-frame .glb files")
    parser.add_argument("--glb_path_with_camera", required=True, help="Path to .glb file containing camera")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--output_video_path", default="output_3d_on_video.mp4", help="Path to save output video")
    parser.add_argument("--sensor_width_mm", type=float, default=36, help="Sensor width in mm")
    parser.add_argument("--sensor_height_mm", type=float, default=24, help="Sensor height in mm")
    parser.add_argument("--focal_length_mm", type=float, default=25, help="Focal length in mm")
    args = parser.parse_args()

    render_3d_overlay_on_video(
        glb_folder=args.glb_folder,
        glb_path_with_camera=args.glb_path_with_camera,
        video_path=args.video_path,
        output_video_path=args.output_video_path,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        focal_length_mm=args.focal_length_mm
    )

if __name__ == "__main__":
    main()
