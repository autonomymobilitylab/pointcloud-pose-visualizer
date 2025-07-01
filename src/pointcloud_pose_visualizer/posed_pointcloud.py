from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Literal, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class PosedPointCloud:
    """
    A point cloud wrapper class with an associated pose and other metadata.
    """

    def __repr__(self):
        return (
            f"PosedPointCloud(created_from={self.created_from},\npose={self.pose},\nscale={self.scale},\nid={self.id})"
        )

    def __init__(
        self,
        pcd: o3d.geometry.PointCloud,
        created_from: Optional[str | PathLike] = "",
        pose: Optional[np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]] = np.eye(4),
        scale: Optional[float] = 1.0,
    ):
        """
        A point cloud with an optional pose.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to be wrapped.
            created_from (str | Path): The source of the point cloud. Used for naming.
            pose (np.ndarray): The pose homogeneous transformation matrix (4x4).
            scale (float): A scale factor for the point cloud.
        """
        self.pcd = pcd
        self.scale = scale
        self.created_from = created_from
        self.name = created_from.stem if isinstance(created_from, Path) else created_from
        self.id = hash(str(self.created_from) + str(datetime.now().timestamp()))  # Unique ID of the point cloud

        if pose is None:
            pose = np.eye(4)
        self.set_pose(pose)
        if scale is None:
            scale = 1.0
        self.set_scale(scale)

    def set_pose(self, pose: np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]):
        """
        Set the pose of the point cloud.

        Args:
            pose (np.ndarray): The new pose homogeneous transformation matrix (4x4).
        """
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix.")
        self.pose = pose
        self.pcd.transform(pose)

    def set_scale(self, scale: float):
        """
        Set the scale of the point cloud.

        Args:
            scale (float): The new scale factor.
        """
        if scale <= 0:
            raise ValueError("Scale must be a positive number.")
        self.scale = scale
        self.pcd.scale(self.scale, center=self.pcd.get_center())


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    """
    Convert sRGB colors to linear RGB.
    """
    # Using the standard sRGB conversion:
    # For values <= 0.04045: linear = color / 12.92
    # For values > 0.04045: linear = ((color + 0.055) / 1.055) ** 2.4
    linear = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)
    return linear


def intensity_to_color(intensities: np.ndarray, colormap: str) -> np.ndarray:
    """
    Normalizes intensities and maps them to colors using a colormap.
    """
    intensity_min = intensities.min()
    intensity_max = intensities.max()
    if intensity_max > intensity_min:
        intensities_norm = (intensities - intensity_min) / (intensity_max - intensity_min)
    else:
        intensities_norm = intensities
    cmap = plt.get_cmap(colormap)
    colors = cmap(intensities_norm)[:, :3]  # Exclude alpha channel if present.
    return colors


def load_posed_pointcloud(filename: str | PathLike, pose=None, scale: Optional[float] = None) -> PosedPointCloud:
    """
    Load point cloud from a file. The file can be in PLY or BIN format.
    """
    filename = Path(filename)
    if filename.suffix == ".ply":
        pcd = load_pcd_from_ply(filename)
    elif filename.suffix == ".bin":
        pcd = load_pcd_from_bin(filename)
    else:
        raise ValueError("Unsupported file format. Only .ply and .bin files are supported.")

    return PosedPointCloud(pcd=pcd, created_from=filename, pose=pose, scale=scale)


def load_pcd_from_bin(filename: str | PathLike, colormap: str = "rainbow") -> o3d.geometry.PointCloud:
    """
    Load point cloud from binary file. The binary file should contain 4 floats per point: x, y, z, intensity
    """
    bin_pcd = np.fromfile(filename, dtype=np.float32)
    pcd_data = bin_pcd.reshape((-1, 4))
    points = pcd_data[:, :3]
    intensities = pcd_data[:, 3]

    colors = intensity_to_color(intensities, colormap)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def load_pcd_from_ply(filename: str | PathLike, convert_srgb: Optional[bool] = True) -> o3d.geometry.PointCloud:
    """
    Load point cloud from a PLY file. Standardizes the point cloud by converting colors to linear
    """
    pcd = o3d.io.read_point_cloud(filename)
    if not pcd.has_points():
        raise ValueError("Point cloud does not contain any points.")

    # Convert colors if they exist and flag is set.
    if pcd.has_colors() and convert_srgb:
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(srgb_to_linear(colors))

    return pcd


def load_pointclouds(filenames: list[str | PathLike]) -> list[PosedPointCloud]:
    """
    Load point clouds from a list of PLY files.
    """
    pointclouds = []
    for file in filenames:
        file = Path(file)
        pointclouds.append(load_posed_pointcloud(file))

    return pointclouds


def load_poses(pose_file: str | PathLike) -> list[np.ndarray]:
    """
    Load poses from a file. The file should contain 4x4 transformation matrices in a text format.
    """
    pose_file = Path(pose_file)
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file '{pose_file}' does not exist.")

    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            if line.strip():
                homogeneous_matrix = np.fromstring(line.strip(), sep=" ").reshape(4, 4)
                poses.append(homogeneous_matrix)

    return poses


def load_pointcloud_files(pointclouds):
    supported_filetypes = [".ply", ".bin"]
    pcl_files = []
    for pcl_arg in pointclouds:
        pcl_path = Path(pcl_arg)
        if not pcl_path.exists():
            raise click.FileError(pcl_arg, hint="Could not open file or folder")
        if pcl_path.is_dir():
            for extension in supported_filetypes:
                pcl_files.extend(sorted(pcl_path.glob(f"*{extension}")))
            if len(pcl_files) == 0:
                raise click.FileError(str(pcl_path), hint="No point clouds found in the folder")
            print(f"Found {len(pcl_files)} point clouds in {pcl_path}")
        else:
            if pcl_path.suffix not in supported_filetypes:
                raise click.BadParameter(f"Only {', '.join(supported_filetypes)} files are supported")
            pcl_files.append(pcl_path)
    return pcl_files
