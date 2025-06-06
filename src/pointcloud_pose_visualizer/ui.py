import platform
from typing import Literal

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation

from pointcloud_pose_visualizer.callbacks import PointCloudUIController
from pointcloud_pose_visualizer.posed_pointcloud import PosedPointCloud
from pointcloud_pose_visualizer.utils import hom2rot, hom2transl, rot2hom


# Menu item IDs
MENU_PCL_OPEN = 1
MENU_DIR_OPEN = 2
MENU_POSE_OPEN = 3
MENU_EXPORT = 4
MENU_QUIT = 5
MENU_SHOW_SETTINGS = 11
MENU_SHOW_INFOBAR = 12
MENU_ABOUT = 21


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


class AppState:
    def __init__(self):
        self.pointclouds = []
        self.poses = []
        # Kept for future use, if needed
        # self.toggle_show_coords = False
        # self.toggle_show_pointcloud = True


class PointCloudTransformUI:
    """
    A simple GUI for visualizing and transforming point clouds using Open3D.
    """

    def __init__(
        self,
        window,
        pointclouds: list[PosedPointCloud] = [],
        poses: list[np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]] = [],
    ):
        self.settings = Settings()
        self.state = AppState()

        self.state.poses = []
        self.state.pointclouds = pointclouds

        self.controller = PointCloudUIController(self)

        # Create a window
        self.window = window
        self.setup_ui()

        # Load point clouds
        for i in range(len(self.state.pointclouds)):
            self.add_pointcloud_to_scene(self.state.pointclouds[i], flip=self.settings.flip)

        if poses:
            self.state.poses = poses
            self.update_poses(self.state.pointclouds, poses)

        self.refresh_infobar()
        self.reset_camera_view()

    def setup_ui(self):
        em = self.window.theme.font_size
        margin = 0.5 * em

        isMacOS = platform.system() == "Darwin"

        # Create a SceneWidget for 3D rendering.
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(self.settings.background_color)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = self.settings.point_size_default

        # Disable LOD downsampling to avoid culling issues (hopefully not needed).
        # self.scene.scene.downsample_threshold = 0

        # -------------- Slider panel --------------
        # Create the vertical slider panel.
        self.panel = gui.Vert(0.05 * em, gui.Margins(margin, margin, margin, 2 * margin))
        self.panel.add_child(gui.Label("Point cloud settings:"))

        # Point cloud selector
        self.dropdown = gui.Combobox()
        for pointcloud in self.state.pointclouds:
            self.dropdown.add_item(pointcloud.name)
        self.dropdown.selected_index = 0
        self.dropdown.set_on_selection_changed(self.controller.on_dropdown_changed)
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
            slider.set_on_value_changed(self.controller.on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider

        # Translation sliders
        for slider_key in ["tx", "ty", "tz"]:
            self.panel.add_child(gui.Label(slider_labels[slider_key]))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.settings.translation_limit, self.settings.translation_limit)
            slider.double_value = self.settings.translation_default
            slider.set_on_value_changed(self.controller.on_transform_slider_changed)
            self.panel.add_child(slider)
            self.sliders[slider_key] = slider

        # Point cloud scale slider
        self.panel.add_child(gui.Label(slider_labels["scale"]))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(0.01, self.settings.scaling_limit)
        slider.double_value = self.settings.scale_default
        slider.set_on_value_changed(self.controller.on_transform_slider_changed)
        self.panel.add_child(slider)
        self.sliders["scale"] = slider

        self.panel.add_fixed(0.5 * em)

        # Toggle showing point cloud
        self.toggle_show_pointcloud_button = gui.ToggleSwitch("Show point cloud")
        self.toggle_show_pointcloud_button.is_on = True
        self.toggle_show_pointcloud_button.set_on_clicked(self.controller.on_toggle_show_pointcloud)
        self.panel.add_child(self.toggle_show_pointcloud_button)

        self.panel.add_fixed(0.5 * em)

        remove_button = gui.Button("Remove pointcloud")
        remove_button.set_on_clicked(self.controller.on_remove_current_pointcloud)
        self.panel.add_child(remove_button)

        self.panel.add_fixed(0.5 * em)
        self.panel.add_child(gui.Label("Global settings:"))

        # Point size slider
        self.panel.add_child(gui.Label(slider_labels["pointsize"]))
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(1, self.settings.point_size_limit)
        slider.double_value = self.mat.point_size
        slider.set_on_value_changed(self.controller.on_point_size_slider)
        self.panel.add_child(slider)
        self.sliders["pointsize"] = slider

        self.panel.add_fixed(0.5 * em)

        # Print button
        print_button = gui.Button("Print TF")
        print_button.set_on_clicked(self.controller.on_print_tf)
        # Reset button
        reset_button = gui.Button("Reset view")
        reset_button.set_on_clicked(self.controller.on_reset)
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
        self.toggle_show_coords_button = gui.ToggleSwitch("Show Coordinate Axes")
        self.toggle_show_coords_button.set_on_clicked(self.controller.on_toggle_coordinate_axes)
        self.panel.add_child(self.toggle_show_coords_button)

        self.panel.add_fixed(0.5 * em)

        # Create a color picker panel
        background_color_picker = gui.ColorEdit()
        background_color_picker.color_value = gui.Color(*self.settings.background_color)
        background_color_picker.set_on_value_changed(self.controller.on_background_color_changed)

        self.panel.add_child(gui.Label("Background color:"))
        self.panel.add_child(background_color_picker)
        # ------------ End Slider panel ------------

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open point cloud...", MENU_PCL_OPEN)
            file_menu.add_item("Open point cloud directory...", MENU_DIR_OPEN)
            file_menu.add_item("Load pose file...", MENU_POSE_OPEN)
            file_menu.add_item("Export Current Image...", MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Show settings panel", MENU_SHOW_SETTINGS)
            settings_menu.set_checked(MENU_SHOW_SETTINGS, True)
            settings_menu.add_item("Show info bar", MENU_SHOW_INFOBAR)
            settings_menu.set_checked(MENU_SHOW_INFOBAR, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", MENU_ABOUT)

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
        self.window.set_on_menu_item_activated(MENU_PCL_OPEN, self.controller.on_menu_pcl_open)
        self.window.set_on_menu_item_activated(MENU_DIR_OPEN, self.controller.on_menu_pcl_dir_open)
        self.window.set_on_menu_item_activated(MENU_POSE_OPEN, self.controller.on_menu_pose_open)
        self.window.set_on_menu_item_activated(MENU_EXPORT, self.controller.on_menu_export)
        self.window.set_on_menu_item_activated(MENU_QUIT, self.controller.on_menu_quit)
        self.window.set_on_menu_item_activated(
            MENU_SHOW_SETTINGS,
            self.controller.on_menu_toggle_settings_panel,
        )
        self.window.set_on_menu_item_activated(MENU_SHOW_INFOBAR, self.controller.on_menu_toggle_infobar)
        self.window.set_on_menu_item_activated(MENU_ABOUT, self.controller.on_menu_about)
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

        self.set_sliders_to_pose(self.state.pointclouds[self.dropdown.selected_index])

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
        if len(self.state.pointclouds) == 0:
            print("No point clouds loaded. Cannot reset camera view.")
            return
        max_bound = np.max(np.array([pointcloud.pcd.get_max_bound() for pointcloud in self.state.pointclouds]), axis=0)
        min_bound = np.min(np.array([pointcloud.pcd.get_min_bound() for pointcloud in self.state.pointclouds]), axis=0)
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
        if len(self.state.pointclouds) > 0:
            self.infobarlabel.text = "Active point cloud:\n{}".format(
                self.state.pointclouds[self.dropdown.selected_index].created_from
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


def start_gui(
    pointclouds: list[PosedPointCloud] = [], poses: list[np.ndarray[tuple[Literal[4, 4]], np.dtype[np.float64]]] = []
):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Pointcloud alignment tool", 1080, 720)
    PointCloudTransformUI(window, pointclouds, poses)
    gui.Application.instance.run()
