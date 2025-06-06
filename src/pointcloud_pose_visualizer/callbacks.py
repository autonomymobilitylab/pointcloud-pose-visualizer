from datetime import datetime
from pathlib import Path

import numpy as np
import open3d.visualization.gui as gui

from pointcloud_pose_visualizer.posed_pointcloud import load_pointclouds, load_poses
import pointcloud_pose_visualizer.ui as ui
from pointcloud_pose_visualizer.utils import rot_transl2hom, roteuler2rotmat


class PointCloudUIController:
    """Controller class for the PointCloudTransformUI.
    Handles all user interaction with the UI (callbacks).
    """

    def __init__(self, ui):
        self.ui = ui

    # ---- Menu Callbacks ----
    def on_menu_pcl_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", self.ui.window.theme)
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

        dlg.set_on_cancel(self.on_file_dialog_cancel)
        dlg.set_on_done(self.on_single_pointcloud_load_dialog_done)
        self.ui.window.show_dialog(dlg)

    def on_file_dialog_cancel(self):
        self.ui.window.close_dialog()

    def on_single_pointcloud_load_dialog_done(self, filename):
        self.ui.window.close_dialog()

        pointclouds = load_pointclouds([filename])
        self.ui.state.pointclouds.extend(pointclouds)
        for i in range(len(pointclouds)):
            self.ui.add_pointcloud_to_scene(pointclouds[i], flip=self.ui.settings.flip)
        self.ui.reset_camera_view()

    def on_menu_pcl_dir_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose directory to load", self.ui.window.theme)
        dlg.tooltip = "Select a directory containing point cloud files (.ply, .bin)"

        dlg.set_on_cancel(self.on_file_dialog_cancel)
        dlg.set_on_done(self.on_pointcloud_dir_load_dialog_done)
        self.ui.window.show_dialog(dlg)

    def on_pointcloud_dir_load_dialog_done(self, directory):
        self.ui.window.close_dialog()

        # Load all point clouds from the directory
        pcl_files = list(Path(directory).glob("*.ply")) + list(Path(directory).glob("*.bin"))
        if not pcl_files:
            gui.Application.instance.show_message_box(
                "No point clouds found", "No .ply or .bin files found in the directory."
            )
            return

        pointclouds = load_pointclouds(pcl_files)
        self.ui.state.pointclouds.extend(pointclouds)
        for i in range(len(pointclouds)):
            self.ui.add_pointcloud_to_scene(pointclouds[i], flip=self.ui.settings.flip)
            if self.ui.toggle_show_coords_button.is_on:
                self.ui.add_coordinate_frame_to_scene(pointclouds[i])
        self.ui.reset_camera_view()

    def on_menu_pose_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pose file to load", self.ui.window.theme)
        dlg.add_filter(".txt", "Text files (.txt)")
        dlg.tooltip = "Select a text file containing poses"

        dlg.set_on_cancel(self.on_file_dialog_cancel)
        dlg.set_on_done(self.on_pose_load_dialog_done)
        self.ui.window.show_dialog(dlg)

    def on_pose_load_dialog_done(self, filename):
        self.ui.window.close_dialog()
        poses = load_poses(filename)
        if len(self.ui.state.pointclouds) != len(poses):
            gui.Application.instance.show_message_box(
                "Wrong amount of poses",
                f"Number of poses ({len(poses)}) does not match number of point clouds ({len(self.ui.state.pointclouds)}).",
            )
            return
        self.ui.update_poses(self.ui.state.pointclouds, poses)
        self.ui.reset_camera_view()

    def on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.ui.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        if len(self.ui.state.pointclouds) > 3:
            try:
                pcl_names = f"{self.ui.state.pointclouds[0].created_from.parent}_"
            except Exception:
                pcl_names = "{}_" * len(self.ui.state.pointclouds)
                pcl_names = pcl_names.format(*[pointcloud.name for pointcloud in self.ui.state.pointclouds[:3]])
        else:
            pcl_names = "{}_" * len(self.ui.state.pointclouds)
            pcl_names = pcl_names.format(*[pointcloud.name for pointcloud in self.ui.state.pointclouds])
        # set_path("filename") defines a default filename for the saved filed in the dialog
        dlg.set_path(f"pcls_{pcl_names}{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        dlg.set_on_cancel(self.on_file_dialog_cancel)
        dlg.set_on_done(self.on_export_dialog_done)
        self.ui.window.show_dialog(dlg)

    def on_export_dialog_done(self, filename):
        self.ui.window.close_dialog()
        self.ui.export_image(filename)

    def on_menu_quit(self):
        gui.Application.instance.quit()

    def on_menu_toggle_settings_panel(self):
        self.ui.panel.visible = not self.ui.panel.visible
        gui.Application.instance.menubar.set_checked(ui.MENU_SHOW_SETTINGS, self.ui.panel.visible)

    def on_menu_toggle_infobar(self):
        self.ui.infobar.visible = not self.ui.infobar.visible
        gui.Application.instance.menubar.set_checked(ui.MENU_SHOW_INFOBAR, self.ui.infobar.visible)

    def on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.ui.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI for point cloud alignment"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self.on_about_ok)

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
        self.ui.window.show_dialog(dlg)

    def on_about_ok(self):
        self.ui.window.close_dialog()

    # ---- End Menu Callbacks ----

    # ---- Panel Callbacks ----
    def on_dropdown_changed(self, text, index):
        """
        Callback for when the dropdown selection changes.
        Updates the ui.sliders to match the selected point cloud's pose.
        """
        selected_pcd = self.ui.state.pointclouds[index]
        self.ui.set_sliders_to_pose(selected_pcd)
        self.ui.refresh_infobar()
        if self.ui.scene.scene.geometry_is_visible(str(selected_pcd.id)):
            self.ui.toggle_show_pointcloud_button.is_on = True
        else:
            self.ui.toggle_show_pointcloud_button.is_on = False

    def on_transform_slider_changed(self, new_value):
        """
        Callback for when any of the transformation ui.sliders change.
        Updates the transformation of the currently selected point cloud.
        """
        # Retrieve slider values.
        x_deg = self.ui.sliders["rx"].double_value
        y_deg = self.ui.sliders["ry"].double_value
        z_deg = self.ui.sliders["rz"].double_value
        scale = self.ui.sliders["scale"].double_value
        trans_x = self.ui.sliders["tx"].double_value
        trans_y = self.ui.sliders["ty"].double_value
        trans_z = self.ui.sliders["tz"].double_value

        # Create transformation matrix with scaling and translation
        R = roteuler2rotmat(x_deg, y_deg, z_deg, units="deg")
        homogeneous_transform = rot_transl2hom(scale * R, [trans_x, trans_y, trans_z])

        # Update point cloud transformation.
        if len(self.ui.state.pointclouds) == 0:
            return
        active_pointcloud = self.ui.state.pointclouds[self.ui.dropdown.selected_index]
        active_pointcloud.set_pose(homogeneous_transform)
        self.ui.scene.scene.set_geometry_transform(f"{active_pointcloud.id}", homogeneous_transform)
        if self.ui.scene.scene.has_geometry(f"{active_pointcloud.id}_coordinate_frame"):
            self.ui.scene.scene.set_geometry_transform(
                f"{active_pointcloud.id}_coordinate_frame", homogeneous_transform
            )
        self.ui.scene.force_redraw()

    def on_toggle_show_pointcloud(self, is_on):
        """Toggle the visibility of the currently active point cloud in the scene."""
        if len(self.ui.state.pointclouds) == 0:
            return
        active_pointcloud = self.ui.state.pointclouds[self.ui.dropdown.selected_index]
        active_pointcloud_coordframe = f"{active_pointcloud.id}_coordinate_frame"
        if is_on:
            self.ui.scene.scene.show_geometry(str(active_pointcloud.id), show=True)
            if self.ui.toggle_show_coords_button.is_on:
                if self.ui.scene.scene.has_geometry(active_pointcloud_coordframe):
                    self.ui.scene.scene.show_geometry(active_pointcloud_coordframe, show=True)
                else:
                    self.ui.add_coordinate_frame_to_scene(active_pointcloud)
        else:
            self.ui.scene.scene.show_geometry(str(active_pointcloud.id), show=False)
            if self.ui.scene.scene.has_geometry(active_pointcloud_coordframe):
                self.ui.scene.scene.show_geometry(active_pointcloud_coordframe, show=False)
        self.ui.scene.force_redraw()

    def on_remove_current_pointcloud(self):
        """Remove the currently selected point cloud from the scene."""
        if len(self.ui.state.pointclouds) == 0:
            return
        selected_index = self.ui.dropdown.selected_index
        active_pointcloud = self.ui.state.pointclouds[selected_index]
        self.ui.scene.scene.remove_geometry(str(active_pointcloud.id))
        if self.ui.scene.scene.has_geometry(f"{active_pointcloud.id}_coordinate_frame"):
            self.ui.scene.scene.remove_geometry(f"{active_pointcloud.id}_coordinate_frame")
        self.ui.dropdown.remove_item(selected_index)
        del self.ui.state.pointclouds[selected_index]
        if len(self.ui.state.pointclouds) > 0:
            self.ui.dropdown.selected_index = min(selected_index, len(self.ui.state.pointclouds) - 1)
            self.ui.set_sliders_to_pose(self.ui.state.pointclouds[self.ui.dropdown.selected_index])
            if self.ui.scene.scene.geometry_is_visible(
                str(self.ui.state.pointclouds[self.ui.dropdown.selected_index].id)
            ):
                self.ui.toggle_show_pointcloud_button.is_on = True
            else:
                self.ui.toggle_show_pointcloud_button.is_on = False
        self.ui.refresh_infobar()

    def on_toggle_coordinate_axes(self, is_on):
        """Toggle the visibility of coordinate axes in the scene."""
        for pointcloud in self.ui.state.pointclouds:
            if self.ui.scene.scene.geometry_is_visible(str(pointcloud.id)):
                if is_on:
                    if not self.ui.scene.scene.has_geometry(f"{pointcloud.id}_coordinate_frame"):
                        self.ui.add_coordinate_frame_to_scene(pointcloud)
                else:
                    if self.ui.scene.scene.has_geometry(f"{pointcloud.id}_coordinate_frame"):
                        self.ui.scene.scene.remove_geometry(f"{pointcloud.id}_coordinate_frame")

    def on_point_size_slider(self, size):
        self.ui.mat.point_size = int(size)
        self.ui.scene.scene.update_material(self.ui.mat)

    def on_print_tf(self):
        """Print the transformation matrix and scale of the currently selected point cloud."""
        if len(self.ui.state.pointclouds) == 0:
            print("No point clouds loaded.")
            return
        with np.printoptions(suppress=True):
            print("Transformation matrix (with scale):")
            print(self.ui.state.pointclouds[self.ui.dropdown.selected_index].pose)
            print(f"Scale: {self.ui.sliders['scale'].double_value}")
            # T = np.eye(4)
            # T[:3, :3] = R
            # T[:3, 3] = homogeneous_transform[:3, 3]
            # print("Transformation matrix without scale:")
            # print(T)

    def on_reset(self):
        """Reset camera view."""
        self.ui.scene.scene.camera.look_at(*self.ui.settings.camera_view)
        self.ui.scene.force_redraw()

    def on_background_color_changed(self, new_color):
        # Update material color
        self.ui.scene.scene.set_background([new_color.red, new_color.green, new_color.blue, 1.0])

    # ---- End Panel Callbacks ----
