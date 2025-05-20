from datetime import datetime
import math
from pathlib import Path
import platform
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation

from utils import rot2hom, rot_transl2hom, hom2rot, hom2transl


isMacOS = platform.system() == "Darwin"


class Settings:
    """
    Default settings for the point cloud transformation UI.
    """

    # Rotation defaults
    rotation_default = 0  # (in degrees)

    # Scale default
    scale_default = 1.0

    # Translation defaults
    translation_default = 0.0

    # Point size default
    point_size_default = 2

    # Limits
    rotation_limit = 180
    scaling_limit = 3.0
    translation_limit = 50.0
    point_size_limit = 6

    # Flip point clouds
    flip = True

    #

    # Camera view
    camera_view = [
        [0, 0, 0],
        [-60, -60, 60],
        [0, 0, 1],
    ]  # View direction, eye position, up direction


class PointCloudTransformUI:
    """
    A simple GUI for visualizing and transforming point clouds using Open3D.
    """

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_INFOBAR = 12
    MENU_ABOUT = 21

    def __init__(
        self,
        window,
        ply_files: list[str | Path],
        initial_transform: np.ndarray = np.eye(4),
    ):
        self.window = window
        em = self.window.theme.font_size
        margin = 0.25 * em

        self.settings = Settings()

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

        if initial_transform.size != 16:
            raise ValueError("Initial transformation matrix must be a 4x4 matrix.")
        initial_transform = initial_transform.reshape(4, 4)
        self.initial_transform = initial_transform

        (
            self.settings.rotation_default,
            self.settings.rotation_default,
            self.settings.rotation_default,
        ) = Rotation.from_matrix(hom2rot(self.initial_transform)).as_euler("xyz", degrees=True)
        (
            self.settings.translation_default,
            self.settings.translation_default,
            self.settings.translation_default,
        ) = hom2transl(self.initial_transform)

        # Needed for printing transformation matrix
        self.homogeneous_transform = self.initial_transform
        self.R = np.eye(3)

        # Create a SceneWidget for 3D rendering.
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        # self.scene.scene.set_background([255, 255, 255, 1])
        self.scene.scene.set_background([110, 0, 0, 1])
        # self.scene.scene.set_background([0, 0, 255, 1])

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = self.settings.point_size_default

        self.pointclouds, self.pointcloud_names = self.load_pointclouds(ply_files)
        for i in range(len(self.pointclouds)):
            self.add_pcl_to_scene(self.pointclouds[i], str(self.pointcloud_names[i]), flip=self.settings.flip)

        self.reset_camera_view()

        # Show coordinate axes
        # self.scene.scene.show_axes(True)

        # Disable LOD downsampling to avoid culling issues (hopefully not needed).
        # self.scene.scene.downsample_threshold = 0

        # -------------- Slider panel --------------
        # Create the vertical slider panel.
        self.panel = gui.Vert(0.05 * em, gui.Margins(margin, margin, margin, 2 * margin))

        self.sliders = {}
        for slider_key in ["rx", "ry", "rz"]:
            self.panel.add_child(gui.Label(slider_labels[slider_key]))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.settings.rotation_limit, self.settings.rotation_limit)
            slider.double_value = self.settings.rotation_default
            slider.set_on_value_changed(self._on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider
        for slider_key in ["tx", "ty", "tz"]:
            self.panel.add_child(gui.Label(slider_labels[slider_key]))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.settings.translation_limit, self.settings.translation_limit)
            slider.double_value = self.settings.translation_default
            slider.set_on_value_changed(self._on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider

        self.panel.add_child(gui.Label(slider_labels["scale"]))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(0.01, self.settings.scaling_limit)
        slider.double_value = self.settings.scale_default
        slider.set_on_value_changed(self._on_transform_slider_changed)
        self.panel.add_child(slider)
        self.sliders["scale"] = slider

        self.panel.add_child(gui.Label(slider_labels["pointsize"]))
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(1, self.settings.point_size_limit)
        slider.double_value = self.mat.point_size
        slider.set_on_value_changed(self._on_point_size_slider)
        self.panel.add_child(slider)
        self.sliders["pointsize"] = slider

        # Print button
        print_button = gui.Button("Print TF")
        print_button.set_on_clicked(self._on_print_tf)
        # Reset button
        reset_button = gui.Button("Reset TF")
        reset_button.set_on_clicked(self._on_reset)
        # Add buttons side by side
        button_bar = gui.Horiz()
        button_bar.add_stretch()
        button_bar.add_child(print_button)
        button_bar.add_fixed(0.5 * em)
        button_bar.add_child(reset_button)
        button_bar.add_stretch()
        self.panel.add_child(button_bar)
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
            file_menu.add_item("Open...", PointCloudTransformUI.MENU_OPEN)
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
        window.set_on_menu_item_activated(PointCloudTransformUI.MENU_OPEN, self._on_menu_open)
        window.set_on_menu_item_activated(PointCloudTransformUI.MENU_EXPORT, self._on_menu_export)
        window.set_on_menu_item_activated(PointCloudTransformUI.MENU_QUIT, self._on_menu_quit)
        window.set_on_menu_item_activated(
            PointCloudTransformUI.MENU_SHOW_SETTINGS,
            self._on_menu_toggle_settings_panel,
        )
        window.set_on_menu_item_activated(PointCloudTransformUI.MENU_SHOW_INFOBAR, self._on_menu_toggle_infobar)
        window.set_on_menu_item_activated(PointCloudTransformUI.MENU_ABOUT, self._on_menu_about)
        # ---- End Menu ----

        # ---- Info bar ----
        self.infobar = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0))
        infobartext = "Point clouds:\n{}".format(*[name for name in self.pointcloud_names])
        self.infobar.add_child(gui.Label(infobartext))
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
    def _on_menu_open(self):
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

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()

        pointclouds, pointcloud_names = self.load_pointclouds([filename])
        self.pointclouds.extend(pointclouds)
        self.pointcloud_names.extend(pointcloud_names)
        for i in range(len(pointclouds)):
            self.add_pcl_to_scene(pointclouds[i], str(pointcloud_names[i]), flip=self.settings.flip)
        self.reset_camera_view()

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        pcl_names = "{}_" * len(self.pointcloud_names)
        pcl_names = pcl_names.format(*[name.stem for name in self.pointcloud_names])
        # set_path() defines a default filename for the dialog
        dlg.set_path(f"pcls_{pcl_names}{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self.scene.frame
        self.export_image(filename, frame.width, frame.height)

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
    def _on_transform_slider_changed(self, new_value):
        # Retrieve slider values.
        x_deg = self.sliders["rx"].double_value
        y_deg = self.sliders["ry"].double_value
        z_deg = self.sliders["rz"].double_value
        x_rad = math.radians(x_deg)
        y_rad = math.radians(y_deg)
        z_rad = math.radians(z_deg)

        scale = self.sliders["scale"].double_value
        trans_x = self.sliders["tx"].double_value
        trans_y = self.sliders["ty"].double_value
        trans_z = self.sliders["tz"].double_value

        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(x_rad), -math.sin(x_rad)],
                [0, math.sin(x_rad), math.cos(x_rad)],
            ]
        )
        Ry = np.array(
            [
                [math.cos(y_rad), 0, math.sin(y_rad)],
                [0, 1, 0],
                [-math.sin(y_rad), 0, math.cos(y_rad)],
            ]
        )
        Rz = np.array(
            [
                [math.cos(z_rad), -math.sin(z_rad), 0],
                [math.sin(z_rad), math.cos(z_rad), 0],
                [0, 0, 1],
            ]
        )
        # Combine rotations: order X, Y, Z.
        self.R = Rz @ Ry @ Rx

        # Create transformation matrix with scaling and translation
        self.homogeneous_transform = rot_transl2hom(scale * self.R, [trans_x, trans_y, trans_z])

        # Update point cloud transformation.
        self.scene.scene.set_geometry_transform(f"{str(self.pointcloud_names[-1])}", self.homogeneous_transform)
        self.scene.force_redraw()

    def _on_print_tf(self):
        with np.printoptions(suppress=True):
            print("Transformation matrix:")
            print(self.homogeneous_transform)
            print(f"Scale: {self.sliders['scale'].double_value}")
            T = np.eye(4)
            T[:3, :3] = self.R
            T[:3, 3] = self.homogeneous_transform[:3, 3]
            print("Transformation matrix without scale:")
            print(T)

    def _on_reset(self):
        """Reset transformation sliders to their default values."""
        self.sliders["rx"].double_value = self.settings.rotation_default
        self.sliders["ry"].double_value = self.settings.rotation_default
        self.sliders["rz"].double_value = self.settings.rotation_default
        self.sliders["scale"].double_value = self.settings.scale_default
        self.sliders["tx"].double_value = self.settings.translation_default
        self.sliders["ty"].double_value = self.settings.translation_default
        self.sliders["tz"].double_value = self.settings.translation_default
        self._on_transform_slider_changed(0)
        self.scene.scene.camera.look_at(*self.settings.camera_view)
        self.scene.force_redraw()

    def _on_point_size_slider(self, size):
        self.mat.point_size = int(size)
        self.scene.scene.update_material(self.mat)

    # ---- End Panel Callbacks ----

    def export_image(self, path, width, height):
        def on_image(image):
            img = image
            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self.scene.scene.scene.render_to_image(on_image)

    def intensity_to_color(self, intensities: np.ndarray, colormap: str) -> np.ndarray:
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

    def load_pointcloud(self, filename: str | Path) -> o3d.geometry.PointCloud:
        """
        Load point cloud from a file. The file can be in PLY or BIN format.
        """
        filename = Path(filename)
        if filename.suffix == ".ply":
            return self.load_pcl_from_ply(filename)
        elif filename.suffix == ".bin":
            return self.load_pcl_from_bin(filename)
        else:
            raise ValueError("Unsupported file format. Only .ply and .bin files are supported.")

    def load_pcl_from_bin(self, filename: str | Path, colormap: str = "rainbow") -> o3d.geometry.PointCloud:
        """
        Load point cloud from binary file. The binary file should contain 4 floats per point: x, y, z, intensity
        """
        bin_pcd = np.fromfile(filename, dtype=np.float32)
        pcd_data = bin_pcd.reshape((-1, 4))
        points = pcd_data[:, :3]
        intensities = pcd_data[:, 3]

        colors = self.intensity_to_color(intensities, colormap)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def srgb_to_linear(self, colors):
        # Using the standard sRGB conversion:
        # For values <= 0.04045: linear = color / 12.92
        # For values > 0.04045: linear = ((color + 0.055) / 1.055) ** 2.4
        colors = np.asarray(colors)
        linear = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)
        return linear

    def load_pcl_from_ply(self, filename: str | Path, convert_srgb: bool = True) -> o3d.geometry.PointCloud:
        """
        Load point cloud from a PLY file. Standardizes the point cloud by converting colors to linear
        """
        pcd = o3d.io.read_point_cloud(filename)
        if not pcd.has_points():
            raise ValueError("Point cloud does not contain any points.")

        # Convert colors if they exist and flag is set.
        if pcd.has_colors() and convert_srgb:
            colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(self.srgb_to_linear(colors))

        return pcd

    def load_pointclouds(
        self, filenames: list[str | Path], convert_srgb: bool = True
    ) -> Tuple[list[o3d.geometry.PointCloud], list[Path]]:
        """
        Load point clouds from a list of PLY files.
        """
        # if len(filenames) > 2:
        #     raise ValueError("Max two point clouds can be visualized at a time.")

        pointclouds = []
        pointcloud_names = []
        for file in filenames:
            file = Path(file)
            pointclouds.append(self.load_pointcloud(file))
            pointcloud_names.append(file)

        return pointclouds, pointcloud_names

    def add_pcl_to_scene(
        self, pointcloud: o3d.geometry.PointCloud, pointcloud_name: str, pose: np.ndarray = None, flip: bool = False
    ):
        """
        Add a point cloud to the scene in the given pose (or to the center of the scene if no pose is given).

        Args:
            pointcloud (o3d.geometry.PointCloud): The point cloud to add.
            pointcloud_name (str): The name of the point cloud.
            pose (np.ndarray, optional): The pose transformation to apply to the point cloud. Defaults to None.
            flip (bool, optional): Whether to flip the point cloud upside down. Defaults to False.
        """
        center = pointcloud.get_center()
        pointcloud.translate(-center)
        if pose is not None:
            pointcloud.transform(pose)
        if flip:
            pointcloud.transform(rot2hom(Rotation.from_euler("xz", [-90, -90], degrees=True)))
        self.scene.scene.add_geometry(f"{pointcloud_name}", pointcloud, self.mat)

    def reset_camera_view(self):
        # Setup camera based on the largest point cloud's bounding box.
        max_bound = max([max(pcl.get_max_bound()) for pcl in self.pointclouds])
        self.settings.camera_view = [
            [0, 0, 0],
            [-max_bound, max_bound, max_bound],
            [0, 0, 1],
        ]  # View direction, eye position, up direction
        self.scene.scene.camera.look_at(*self.settings.camera_view)

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


def main(pcl_files, initial_transform=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Pointcloud alignment tool", 1920, 1080)
    if initial_transform is not None:
        PointCloudTransformUI(window, pcl_files, initial_transform)
    else:
        PointCloudTransformUI(window, pcl_files)
    gui.Application.instance.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Pointcloud GUI",
        description="GUI for visualizing and transforming point clouds using Open3D",
    )
    parser.add_argument(
        "pointclouds",
        type=str,
        nargs="+",
        help="Path of one or more point cloud files.",
    )
    parser.add_argument("-t", "--initial-transform", type=str, help="Initial transformation matrix")
    args = parser.parse_args()
    pcl_files = []
    for sysarg in sys.argv[1:]:
        if not Path.exists(Path(sysarg)):
            raise FileNotFoundError("Could not open file '" + sysarg + "'")
        if Path(sysarg).suffix not in [".ply", ".bin"]:
            raise ValueError("Only .ply and .bin files are supported")
        pcl_files.append(sysarg)
    initial_transform = np.loadtxt(args.initial_transform) if args.initial_transform else None

    main(pcl_files, initial_transform)
