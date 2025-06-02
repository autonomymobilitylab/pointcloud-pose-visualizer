from datetime import datetime
from pathlib import Path
import platform
from typing import Literal

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation

from utils import hom2rot, hom2transl, rot2hom, rot_transl2hom, roteuler2rotmat


isMacOS = platform.system() == "Darwin"


class Settings:
    """
    Default settings for the point cloud transformation UI.
    """

    translation_default = 0.0
    rotation_default = 0  # (in degrees)
    scale_default = 1.0
    point_size_default = 2

    rotation_limit = 180  # min and max (in degrees)
    scaling_limit = 3.0  # max
    translation_limit = 50.0  # min and max
    point_size_limit = 6  # max

    # Flip point clouds in the scene
    # This is useful for point clouds that are upside down, e.g. from a camera
    flip = True

    background_color = np.array([240, 210, 170, 255]) / 255  # RGBA

    camera_view = [
        [0, 0, 0],
        [-60, -60, 60],
        [0, 0, 1],
    ]  # View direction, eye position, up direction


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
        created_from: str | Path = "",
        pose: np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]] | None = np.eye(4),
        scale: float | None = 1.0,
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


class PointCloudTransformUI:
    """
    A simple GUI for visualizing and transforming point clouds using Open3D.
    """

    MENU_PCL_OPEN = 1
    MENU_DIR_OPEN = 2
    MENU_POSE_OPEN = 3
    MENU_EXPORT = 4
    MENU_QUIT = 5
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_INFOBAR = 12
    MENU_ABOUT = 21

    def __init__(
        self,
        window,
        pointclouds: list[PosedPointCloud] = [],
        poses: list[np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]] = [],
    ):
        self.settings = Settings()

        self.poses = []
        self.pointclouds = pointclouds

        # Create a window
        self.window = window
        self.setup_ui()

        # Load point clouds
        for i in range(len(self.pointclouds)):
            self.add_pointcloud_to_scene(self.pointclouds[i], flip=self.settings.flip)

        if poses:
            self.poses = poses
            self.update_poses(self.pointclouds, poses)

        self.refresh_infobar()
        self.reset_camera_view()

    def setup_ui(self):
        em = self.window.theme.font_size
        margin = 0.5 * em

        # Create a SceneWidget for 3D rendering.
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(self.settings.background_color)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = self.settings.point_size_default

        # Show coordinate axes
        # self.scene.scene.show_axes(True)

        # Disable LOD downsampling to avoid culling issues (hopefully not needed).
        # self.scene.scene.downsample_threshold = 0

        # -------------- Slider panel --------------
        # Create the vertical slider panel.
        self.panel = gui.Vert(0.05 * em, gui.Margins(margin, margin, margin, 2 * margin))
        self.panel.add_child(gui.Label("Point cloud settings:"))

        # Point cloud selector
        self.dropdown = gui.Combobox()
        for pointcloud in self.pointclouds:
            self.dropdown.add_item(pointcloud.name)
        self.dropdown.selected_index = 0
        self.dropdown.set_on_selection_changed(self._on_dropdown_changed)
        self.panel.add_child(self.dropdown)

        slider_labels = {
            "rx": "X Rotation (deg)",
            "ry": "Y Rotation (deg)",
            "rz": "Z Rotation (deg)",
            "tx": "X Translation",
            "ty": "Y Translation",
            "tz": "Z Translation",
            "scale": "Scaling",
            "pointsize": "Point size",
        }
        self.sliders = {}

        # Rotation sliders
        for slider_key in ["rx", "ry", "rz"]:
            self.panel.add_child(gui.Label(slider_labels[slider_key]))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.settings.rotation_limit, self.settings.rotation_limit)
            slider.double_value = self.settings.rotation_default
            slider.set_on_value_changed(self._on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider

        # Translation sliders
        for slider_key in ["tx", "ty", "tz"]:
            self.panel.add_child(gui.Label(slider_labels[slider_key]))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.settings.translation_limit, self.settings.translation_limit)
            slider.double_value = self.settings.translation_default
            slider.set_on_value_changed(self._on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider

        # Point cloud scale slider
        self.panel.add_child(gui.Label(slider_labels["scale"]))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(0.01, self.settings.scaling_limit)
        slider.double_value = self.settings.scale_default
        slider.set_on_value_changed(self._on_transform_slider_changed)
        self.panel.add_child(slider)
        self.sliders["scale"] = slider

        self.panel.add_fixed(0.5 * em)

        # Toggle showing point cloud
        self.toggle_show_pointcloud_button = gui.ToggleSwitch("Show point cloud")
        self.toggle_show_pointcloud_button.is_on = True
        self.toggle_show_pointcloud_button.set_on_clicked(self._on_toggle_show_pointcloud)
        self.panel.add_child(self.toggle_show_pointcloud_button)

        self.panel.add_fixed(0.5 * em)

        remove_button = gui.Button("Remove pointcloud")
        remove_button.set_on_clicked(self._on_remove_current_pointcloud)
        self.panel.add_child(remove_button)

        self.panel.add_fixed(0.5 * em)
        self.panel.add_child(gui.Label("Global settings:"))

        # Point size slider
        self.panel.add_child(gui.Label(slider_labels["pointsize"]))
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(1, self.settings.point_size_limit)
        slider.double_value = self.mat.point_size
        slider.set_on_value_changed(self._on_point_size_slider)
        self.panel.add_child(slider)
        self.sliders["pointsize"] = slider

        self.panel.add_fixed(0.5 * em)

        # Print button
        print_button = gui.Button("Print TF")
        print_button.set_on_clicked(self._on_print_tf)
        # Reset button
        reset_button = gui.Button("Reset view")
        reset_button.set_on_clicked(self._on_reset)
        # Add buttons side by side
        button_bar = gui.Horiz()
        button_bar.add_stretch()
        button_bar.add_child(print_button)
        button_bar.add_fixed(0.5 * em)
        button_bar.add_child(reset_button)
        button_bar.add_stretch()
        self.panel.add_child(button_bar)

        self.panel.add_fixed(0.5 * em)

        # Toggle coordinate axes
        self.toggle_coords_button = gui.ToggleSwitch("Show Coordinate Axes")
        self.toggle_coords_button.set_on_clicked(self._on_toggle_coordinate_axes)
        self.panel.add_child(self.toggle_coords_button)

        self.panel.add_fixed(0.5 * em)

        # Create a color picker panel
        background_color_picker = gui.ColorEdit()
        background_color_picker.color_value = gui.Color(*self.settings.background_color)
        background_color_picker.set_on_value_changed(self._on_background_color_changed)

        self.panel.add_child(gui.Label("Background color:"))
        self.panel.add_child(background_color_picker)
        # ------------ End Slider panel ------------

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", PointCloudTransformUI.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", PointCloudTransformUI.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open point cloud...", PointCloudTransformUI.MENU_PCL_OPEN)
            file_menu.add_item("Open point cloud directory...", PointCloudTransformUI.MENU_DIR_OPEN)
            file_menu.add_item("Load pose file...", PointCloudTransformUI.MENU_POSE_OPEN)
            file_menu.add_item("Export Current Image...", PointCloudTransformUI.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", PointCloudTransformUI.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Show settings panel", PointCloudTransformUI.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(PointCloudTransformUI.MENU_SHOW_SETTINGS, True)
            settings_menu.add_item("Show info bar", PointCloudTransformUI.MENU_SHOW_INFOBAR)
            settings_menu.set_checked(PointCloudTransformUI.MENU_SHOW_INFOBAR, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", PointCloudTransformUI.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_PCL_OPEN, self._on_menu_pcl_open)
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_DIR_OPEN, self._on_menu_pcl_dir_open)
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_POSE_OPEN, self._on_menu_pose_open)
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_EXPORT, self._on_menu_export)
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(
            PointCloudTransformUI.MENU_SHOW_SETTINGS,
            self._on_menu_toggle_settings_panel,
        )
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_SHOW_INFOBAR, self._on_menu_toggle_infobar)
        self.window.set_on_menu_item_activated(PointCloudTransformUI.MENU_ABOUT, self._on_menu_about)
        # ---- End Menu ----

        # ---- Info bar ----
        self.infobar = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0))
        self.infobarlabel = gui.Label("")
        self.refresh_infobar()
        self.infobar.add_child(self.infobarlabel)
        # ---- End Info bar ----

        # Layout the window
        layout = gui.Vert()
        layout.add_child(self.scene)
        layout.add_child(self.infobar)
        self.window.set_on_layout(self.on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.add_child(self.infobar)

    # ---- Menu Callbacks ----
    def _on_menu_pcl_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", self.window.theme)
        dlg.add_filter(".ply, .bin", "Supported files (.ply, .bin)")
        dlg.add_filter("", "All files")
        # Other filters:
        # dlg.add_filter(
        #     ".ply .stl .fbx .obj .off .gltf .glb",
        #     "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, .gltf, .glb)",
        # )
        # dlg.add_filter(
        #     ".xyz .xyzn .xyzrgb .ply .pcd .pts",
        #     "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)",
        # )
        # dlg.add_filter(".ply", "Polygon files (.ply)")
        # dlg.add_filter(".stl", "Stereolithography files (.stl)")
        # dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        # dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        # dlg.add_filter(".off", "Object file format (.off)")
        # dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        # dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        # dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        # dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        # dlg.add_filter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)")
        # dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        # dlg.add_filter(".pts", "3D Points files (.pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_single_pointcloud_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_single_pointcloud_load_dialog_done(self, filename):
        self.window.close_dialog()

        pointclouds = load_pointclouds([filename])
        self.pointclouds.extend(pointclouds)
        for i in range(len(pointclouds)):
            self.add_pointcloud_to_scene(pointclouds[i], flip=self.settings.flip)
        self.reset_camera_view()

    def _on_menu_pcl_dir_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose directory to load", self.window.theme)
        dlg.tooltip = "Select a directory containing point cloud files (.ply, .bin)"

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_pointcloud_dir_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_pointcloud_dir_load_dialog_done(self, directory):
        self.window.close_dialog()

        # Load all point clouds from the directory
        pcl_files = list(Path(directory).glob("*.ply")) + list(Path(directory).glob("*.bin"))
        if not pcl_files:
            gui.Application.instance.show_message_box(
                "No point clouds found", "No .ply or .bin files found in the directory."
            )
            return

        pointclouds = load_pointclouds(pcl_files)
        self.pointclouds.extend(pointclouds)
        for i in range(len(pointclouds)):
            self.add_pointcloud_to_scene(pointclouds[i], flip=self.settings.flip)
            if self.toggle_coords_button.is_on:
                self.add_coordinate_frame_to_scene(pointclouds[i])
        self.reset_camera_view()

    def _on_menu_pose_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pose file to load", self.window.theme)
        dlg.add_filter(".txt", "Text files (.txt)")
        dlg.tooltip = "Select a text file containing poses"

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_pose_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_pose_load_dialog_done(self, filename):
        self.window.close_dialog()
        poses = load_poses(filename)
        if len(self.pointclouds) != len(poses):
            gui.Application.instance.show_message_box(
                "Wrong amount of poses",
                f"Number of poses ({len(poses)}) does not match number of point clouds ({len(self.pointclouds)}).",
            )
            return
        self.update_poses(self.pointclouds, poses)
        self.reset_camera_view()

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        if len(self.pointclouds) > 3:
            try:
                pcl_names = f"{self.pointclouds[0].created_from.parent}_"
            except Exception:
                pcl_names = "{}_" * len(self.pointclouds)
                pcl_names = pcl_names.format(*[pointcloud.name for pointcloud in self.pointclouds[:3]])
        else:
            pcl_names = "{}_" * len(self.pointclouds)
            pcl_names = pcl_names.format(*[pointcloud.name for pointcloud in self.pointclouds])
        # set_path() defines a default filename for the dialog
        dlg.set_path(f"pcls_{pcl_names}{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        self.export_image(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self.panel.visible = not self.panel.visible
        gui.Application.instance.menubar.set_checked(PointCloudTransformUI.MENU_SHOW_SETTINGS, self.panel.visible)

    def _on_menu_toggle_infobar(self):
        self.infobar.visible = not self.infobar.visible
        gui.Application.instance.menubar.set_checked(PointCloudTransformUI.MENU_SHOW_INFOBAR, self.infobar.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI for point cloud alignment"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    # ---- End Menu Callbacks ----

    # ---- Panel Callbacks ----
    def _on_dropdown_changed(self, text, index):
        """
        Callback for when the dropdown selection changes.
        Updates the sliders to match the selected point cloud's pose.
        """
        selected_pcd = self.pointclouds[index]
        self.set_sliders_to_pose(selected_pcd)
        self.refresh_infobar()
        if self.scene.scene.geometry_is_visible(str(selected_pcd.id)):
            self.toggle_show_pointcloud_button.is_on = True
        else:
            self.toggle_show_pointcloud_button.is_on = False

    def _on_transform_slider_changed(self, new_value):
        """
        Callback for when any of the transformation sliders change.
        Updates the transformation of the currently selected point cloud.
        """
        # Retrieve slider values.
        x_deg = self.sliders["rx"].double_value
        y_deg = self.sliders["ry"].double_value
        z_deg = self.sliders["rz"].double_value
        scale = self.sliders["scale"].double_value
        trans_x = self.sliders["tx"].double_value
        trans_y = self.sliders["ty"].double_value
        trans_z = self.sliders["tz"].double_value

        # Create transformation matrix with scaling and translation
        R = roteuler2rotmat(x_deg, y_deg, z_deg, units="deg")
        homogeneous_transform = rot_transl2hom(scale * R, [trans_x, trans_y, trans_z])

        # Update point cloud transformation.
        if len(self.pointclouds) == 0:
            return
        active_pointcloud = self.pointclouds[self.dropdown.selected_index]
        active_pointcloud.set_pose(homogeneous_transform)
        self.scene.scene.set_geometry_transform(f"{active_pointcloud.id}", homogeneous_transform)
        if self.scene.scene.has_geometry(f"{active_pointcloud.id}_coordinate_frame"):
            self.scene.scene.set_geometry_transform(f"{active_pointcloud.id}_coordinate_frame", homogeneous_transform)
        self.scene.force_redraw()

    def _on_toggle_show_pointcloud(self, is_on):
        """Toggle the visibility of the currently active point cloud in the scene."""
        if len(self.pointclouds) == 0:
            return
        active_pointcloud = self.pointclouds[self.dropdown.selected_index]
        active_pointcloud_coordframe = f"{active_pointcloud.id}_coordinate_frame"
        if is_on:
            self.scene.scene.show_geometry(str(active_pointcloud.id), show=True)
            if self.toggle_coords_button.is_on:
                if self.scene.scene.has_geometry(active_pointcloud_coordframe):
                    self.scene.scene.show_geometry(active_pointcloud_coordframe, show=True)
                else:
                    self.add_coordinate_frame_to_scene(active_pointcloud)
        else:
            self.scene.scene.show_geometry(str(active_pointcloud.id), show=False)
            if self.scene.scene.has_geometry(active_pointcloud_coordframe):
                self.scene.scene.show_geometry(active_pointcloud_coordframe, show=False)
        self.scene.force_redraw()

    def _on_remove_current_pointcloud(self):
        """Remove the currently selected point cloud from the scene."""
        if len(self.pointclouds) == 0:
            return
        selected_index = self.dropdown.selected_index
        active_pointcloud = self.pointclouds[selected_index]
        self.scene.scene.remove_geometry(str(active_pointcloud.id))
        if self.scene.scene.has_geometry(f"{active_pointcloud.id}_coordinate_frame"):
            self.scene.scene.remove_geometry(f"{active_pointcloud.id}_coordinate_frame")
        self.dropdown.remove_item(selected_index)
        del self.pointclouds[selected_index]
        if len(self.pointclouds) > 0:
            self.dropdown.selected_index = min(selected_index, len(self.pointclouds) - 1)
            self.set_sliders_to_pose(self.pointclouds[self.dropdown.selected_index])
            if self.scene.scene.geometry_is_visible(str(self.pointclouds[self.dropdown.selected_index].id)):
                self.toggle_show_pointcloud_button.is_on = True
            else:
                self.toggle_show_pointcloud_button.is_on = False
        self.refresh_infobar()

    def _on_toggle_coordinate_axes(self, is_on):
        """Toggle the visibility of coordinate axes in the scene."""
        for pointcloud in self.pointclouds:
            if self.scene.scene.geometry_is_visible(str(pointcloud.id)):
                if is_on:
                    if not self.scene.scene.has_geometry(f"{pointcloud.id}_coordinate_frame"):
                        self.add_coordinate_frame_to_scene(pointcloud)
                else:
                    if self.scene.scene.has_geometry(f"{pointcloud.id}_coordinate_frame"):
                        self.scene.scene.remove_geometry(f"{pointcloud.id}_coordinate_frame")

    def _on_point_size_slider(self, size):
        self.mat.point_size = int(size)
        self.scene.scene.update_material(self.mat)

    def _on_print_tf(self):
        """Print the transformation matrix and scale of the currently selected point cloud."""
        if len(self.pointclouds) == 0:
            print("No point clouds loaded.")
            return
        with np.printoptions(suppress=True):
            print("Transformation matrix (with scale):")
            print(self.pointclouds[self.dropdown.selected_index].pose)
            print(f"Scale: {self.sliders['scale'].double_value}")
            # T = np.eye(4)
            # T[:3, :3] = R
            # T[:3, 3] = homogeneous_transform[:3, 3]
            # print("Transformation matrix without scale:")
            # print(T)

    def _on_reset(self):
        """Reset camera view."""
        self.scene.scene.camera.look_at(*self.settings.camera_view)
        self.scene.force_redraw()

    def _on_background_color_changed(self, new_color):
        # Update material color
        self.scene.scene.set_background([new_color.red, new_color.green, new_color.blue, 1.0])

    # ---- End Panel Callbacks ----

    def export_image(self, path: str):
        def on_image(image):
            img = image
            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self.scene.scene.scene.render_to_image(on_image)

    def update_poses(self, pointclouds: list[PosedPointCloud], poses: list[np.ndarray]):
        """
        Update the poses of the point clouds with a list of new poses.
        The number of poses must match the number of point clouds.
        """
        if len(pointclouds) != len(poses):
            raise ValueError(
                f"Number of poses ({len(poses)}) does not match number of point clouds ({len(pointclouds)})."
            )
        for i, pointcloud in enumerate(pointclouds):
            pointcloud.set_pose(poses[i])
            self.scene.scene.set_geometry_transform(f"{pointcloud.id}", pointcloud.pose)

        self.set_sliders_to_pose(self.pointclouds[self.dropdown.selected_index])

    def add_pointcloud_to_scene(self, pointcloud: PosedPointCloud, flip: bool = False):
        """
        Add a point cloud to the scene in its own pose.

        Args:
            pointcloud (PosedPointCloud): The point cloud to add.
            flip (bool, optional): Whether to flip the point cloud upside down. Defaults to False.
        """
        if flip:
            pointcloud.pcd.transform(rot2hom(Rotation.from_euler("xz", [-90, -90], degrees=True)))
        self.scene.scene.add_geometry(f"{pointcloud.id}", pointcloud.pcd, self.mat)
        self.dropdown.add_item(pointcloud.name)

    def add_coordinate_frame_to_scene(self, pointcloud: PosedPointCloud, size: float = 1.0):
        """Add coordinate axes to the scene for a given point cloud."""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coord_frame.transform(pointcloud.pose)
        self.scene.scene.add_geometry(f"{pointcloud.id}_coordinate_frame", coord_frame, self.mat)

    def set_sliders_to_pose(self, pointcloud: PosedPointCloud):
        """Set the sliders to the pose of the given PosedPointCloud."""
        self.sliders["rx"].double_value, self.sliders["ry"].double_value, self.sliders["rz"].double_value = (
            Rotation.from_matrix(hom2rot(pointcloud.pose)).as_euler("xyz", degrees=True)
        )
        self.sliders["tx"].double_value, self.sliders["ty"].double_value, self.sliders["tz"].double_value = hom2transl(
            pointcloud.pose
        )
        self.sliders["scale"].double_value = pointcloud.scale

    def reset_camera_view(self):
        # Setup camera based on the combined bounding box of all point clouds.
        if len(self.pointclouds) == 0:
            print("No point clouds loaded. Cannot reset camera view.")
            return
        max_bound = np.max(np.array([pointcloud.pcd.get_max_bound() for pointcloud in self.pointclouds]), axis=0)
        min_bound = np.min(np.array([pointcloud.pcd.get_min_bound() for pointcloud in self.pointclouds]), axis=0)
        combined_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        # combined_center = combined_bbox.get_center()
        combined_half_extent = combined_bbox.get_half_extent()
        combined_radius = np.linalg.norm(combined_half_extent)

        self.settings.camera_view = [
            [0, 0, 0],  # View direction (look at)
            combined_radius * np.array([-1, 1, 1]) * 0.7,  # Eye position
            [0, 0, 1],  # Up direction
        ]
        self.scene.scene.camera.look_at(*self.settings.camera_view)

    def refresh_infobar(self):
        """Update the info bar with the currently active point cloud."""
        if len(self.pointclouds) > 0:
            self.infobarlabel.text = "Active point cloud:\n{}".format(
                self.pointclouds[self.dropdown.selected_index].created_from
            )
        else:
            self.infobarlabel.text = "No point clouds loaded."
        # preferred_size = self.infobarlabel.calc_preferred_size(self.window.get_layout_context(), gui.Widget.Constraints())
        # self.infobarlabel.frame = gui.Rect(10, 10, preferred_size.width, preferred_size.height)

    def on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self.scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self.panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height,
        )
        self.panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        self.infobar.frame = gui.Rect(
            r.x,
            r.get_bottom() - 3 * layout_context.theme.font_size,
            r.width,
            3 * layout_context.theme.font_size,
        )
        infobar_width = min(
            r.width,
            self.infobar.calc_preferred_size(layout_context, gui.Widget.Constraints()).width,
        )
        infobar_height = min(
            r.height,
            self.infobar.calc_preferred_size(layout_context, gui.Widget.Constraints()).height,
        )
        self.infobar.frame = gui.Rect(r.x, r.get_bottom() - infobar_height, infobar_width, infobar_height)


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


def load_posed_pointcloud(filename: str | Path, pose=None, scale=None) -> PosedPointCloud:
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


def load_pcd_from_bin(filename: str | Path, colormap: str = "rainbow") -> o3d.geometry.PointCloud:
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


def load_pcd_from_ply(filename: str | Path, convert_srgb: bool = True) -> o3d.geometry.PointCloud:
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


def load_pointclouds(filenames: list[str | Path]) -> list[PosedPointCloud]:
    """
    Load point clouds from a list of PLY files.
    """
    pointclouds = []
    for file in filenames:
        file = Path(file)
        pointclouds.append(load_posed_pointcloud(file))

    return pointclouds


def load_poses(pose_file: str | Path) -> list[np.ndarray]:
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


def start_gui(
    pointclouds: list[PosedPointCloud] = [], poses: list[np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]] = []
):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Pointcloud alignment tool", 1080, 720)
    PointCloudTransformUI(window, pointclouds, poses)
    gui.Application.instance.run()


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
