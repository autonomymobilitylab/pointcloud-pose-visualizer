import click
from pathlib import Path

from pointcloud_pose_visualizer.posed_pointcloud import load_pointcloud_files, load_pointclouds, load_poses
from pointcloud_pose_visualizer.ui import start_gui

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("pointclouds", nargs=-1, required=False, type=click.Path(exists=True))
@click.option(
    "-p", "--pose-file", type=click.Path(exists=True), help="Path of file containing poses for the point clouds."
)
def cli(pointclouds: list[str | Path], pose_file: str | Path):
    """
    GUI for visualizing and transforming point clouds using Open3D.
    """

    pcl_files = load_pointcloud_files(pointclouds)
    posedpointclouds = load_pointclouds(pcl_files)
    poses = load_poses(pose_file) if pose_file else []
    if poses and len(posedpointclouds) != len(poses):
        raise click.BadParameter(
            f"Number of poses ({len(poses)}) does not match number of point clouds ({len(posedpointclouds)})."
        )

    start_gui(posedpointclouds, poses)


if __name__ == "__main__":
    cli()
