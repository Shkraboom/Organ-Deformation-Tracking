import os
import cv2
import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image
from math import atan, degrees, radians
from scipy.spatial.transform import Rotation as R
from pygltflib import GLTF2

os.environ["PYOPENGL_PLATFORM"] = "egl"

def overlay_3d_model_with_projected_points(glb_path, video_path, output_path="overlay_with_points.png", scale_factor=3.534):
    mesh_scene = trimesh.load(glb_path)
    pyrender_scene = pyrender.Scene.from_trimesh_scene(mesh_scene)

    gltf = GLTF2().load(glb_path)
    camera_node_idx = next(i for i, node in enumerate(gltf.nodes) if node.camera is not None)
    camera_node = gltf.nodes[camera_node_idx]

    translation = camera_node.translation or [0, 0, 0]
    rotation_quat = camera_node.rotation or [0, 0, 0, 1]

    camera_position = np.array(translation)
    rotation_matrix = R.from_quat(rotation_quat).as_matrix()

    camera_transform = np.eye(4)
    camera_transform[:3, :3] = rotation_matrix
    camera_transform[:3, 3] = camera_position

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame from video")

    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    sensor_height = 24
    sensor_width = 36
    focal_length_mm = 25
    fov_y = 2 * degrees(atan((sensor_width / 2) / focal_length_mm))

    camera = pyrender.PerspectiveCamera(yfov=radians(fov_y))
    pyrender_scene.add(camera, pose=camera_transform)

    renderer = pyrender.OffscreenRenderer(viewport_width=frame_width, viewport_height=frame_height)
    color, _ = renderer.render(pyrender_scene)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blended = (color.astype(np.float32) * 1 + frame_rgb.astype(np.float32) * 0).astype(np.uint8)

    vertices = np.array(mesh_scene.geometry[list(mesh_scene.geometry.keys())[0]].vertices)
    scaled_vertices = vertices * scale_factor

    fx = frame_width * focal_length_mm / sensor_width
    fy = frame_height * focal_length_mm / sensor_height
    cx = frame_width / 2
    cy = frame_height / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R_cam = rotation_matrix
    tvec = -R_cam @ camera_position.reshape(3, 1)

    projected, _ = cv2.projectPoints(scaled_vertices, R_cam, tvec, K, None)
    projected = projected.reshape(-1, 2).astype(np.int32)

    for x, y in projected:
        if 0 <= x < frame_width and 0 <= y < frame_height:
            cv2.circle(blended, (x, y), 3, (255, 0, 255), -1)

    Image.fromarray(blended).save(output_path)
    print(f"Points added, image saved: {output_path}")
    return projected

def main():
    parser = argparse.ArgumentParser(description="Overlay 3D mesh points on first video frame")
    parser.add_argument("--glb_path", required=True, help="Path to .glb file with embedded camera")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--output", default="overlay_with_points.png", help="Path to output image")
    parser.add_argument("--scale_factor", type=float, default=3.534, help="Scale factor for 3D model")
    args = parser.parse_args()

    overlay_3d_model_with_projected_points(
        glb_path=args.glb_path,
        video_path=args.video_path,
        output_path=args.output,
        scale_factor=args.scale_factor
    )

if __name__ == "__main__":
    main()
