# ##### BEGIN GPL LICENSE BLOCK #####
#
#  idmapper.py , a Blender addon to create an idmap as a vertex color layer.
#  (c) 2016 - 2025 Michel J. Anders (varkenvarken)
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "IDMapper",
    "author": "Michel Anders (varkenvarken)",
    "version": (0, 0, 20250902164500),
    "blender": (4, 4, 0),
    "location": "View3D > Vertex Paint > Paint",
    "description": "Create an idmap as a vertex color layer, grouping related faces by color",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "https://github.com/varkenvarken/IDMapper/issues",
    "category": "Paint",
}

from copy import copy
import numpy as np
import bpy
import bmesh
import blf
from bpy.props import (
    BoolProperty,
    FloatProperty,
    EnumProperty,
    IntProperty,
    FloatVectorProperty,
    StringProperty,
)
from mathutils import Vector, Color
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon as tessellate
from bpy_extras import view3d_utils
from random import random, seed, sample, shuffle
from time import time
from math import sin, cos, log
from collections import defaultdict as dd
import traceback
import csv
from re import match
from os.path import join
from collections import Counter
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .progress import ProgressCM

preview_collections = {}


# Addon prefs
class IDMapperPrefs(bpy.types.AddonPreferences):
    bl_idname = __name__

    allow_duplicate_names: BoolProperty(
        name="Allow duplicates",
        default=False,
        description="Allow duplicate names in Color List",
    )

    undodepth: IntProperty(
        name="Undo Depth",
        default=20,
        min=5,
        max=100,
        description="Maximum number of undo levels in Face Paint mode",
    )

    allselected: BoolProperty(
        name="Work on all selected objects",
        description="Apply IDMapper to all selected mesh objects",
        default=True,
    )

    helptextcolor: FloatVectorProperty(
        name="Help text color",
        description="Color of the help text shown in the lower right corner in face paint mode",
        size=4,
        subtype="COLOR",
        default=[1.0, 1.0, 1.0, 1.0],
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "allow_duplicate_names")
        layout.prop(self, "undodepth")
        layout.prop(self, "allselected")
        layout.prop(self, "helptextcolor")


class IDMapper(bpy.types.Operator):
    bl_idname = "paint.idmapper"
    bl_label = "IDMapper"
    bl_description = "Create an ID-map with distinct colors for related regions"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    similarity: FloatProperty(
        name="Similarity",
        description="How similar adjacent faces should be to be called related (0 = not at all, 1 completely)",
        default=0.9,
        min=0,
        max=1,
        step=1,
        precision=3,
    )
    seams: BoolProperty(
        name="UV Seams", description="Consider edge seams as boundaries", default=False
    )
    sharp: BoolProperty(
        name="Sharp edges",
        description="Consider edge marked as sharp as boundaries",
        default=False,
    )
    bevel_weight: FloatProperty(
        name="Bevel threshold",
        description="Consider edges with a bevel weight above this threshold as boundaries (0 disables)",
        default=0,
        min=0,
        max=1,
    )
    crease_weight: FloatProperty(
        name="Crease threshold",
        description="Consider edges with a crease weight above this threshold as boundaries (0 disables)",
        default=0,
        min=0,
        max=1,
    )
    area_weight: FloatProperty(
        name="Area weight",
        description="Consider face area in similarity test (0=off)",
        default=0,
        min=0,
        max=1,
    )
    smooth: BoolProperty(
        name="Smooth",
        description="Consider faces with a different smooth setting as boundaries",
        default=False,
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed (different values give different colors)",
        default=0,
        min=0,
    )
    merge: BoolProperty(
        name="Merge", description="Merge adjacent isolated faces", default=False
    )
    match: BoolProperty(
        name="Match",
        description="Give regions with the same number of faces the same color",
        default=False,
    )
    matchvar: IntProperty(
        name="Closeness",
        description="How similar the faces of regions with the same number of faces should be (0 = not at all, 7 completely)",
        default=1,
        min=0,
        max=7,
    )
    selected: BoolProperty(
        name="Only selected",
        description="Calculate id map only for selected faces",
        default=False,
    )
    basecolor: FloatVectorProperty(
        name="Base color",
        description="Color assigned to non-selected faces or faces with an mepty material slot",
        subtype="COLOR",
        size=4,
        default=Vector((0, 0, 0, 1)),
    )
    number_of_ids: BoolProperty(
        name="Number of ids",
        description="Limit the number of distinct vertex colors assigned",
        default=False,
    )
    colors: IntProperty(
        name="Colors",
        description="Maximum number of distinct vertex colors to use",
        default=8,
        min=1,
        max=256,
    )

    method: EnumProperty(
        items=[
            (
                "HEURISTIC",
                "Heuristic",
                "Use edge and face characteristics to identify similar areas",
            ),
            ("MATERIALID", "Materialid", "Use face material ids to identify areas"),
            ("FACEMAPID", "Facemapid", "Use facemaps membership to identify areas"),
        ],
        name="method",
        description="Method to define areas",
        default="HEURISTIC",
    )

    # materialid        : BoolProperty          (name="By Material id",     description="Calculate id map based on assigned materials (disabled when no materials present on mesh)", default=False)
    # facemapid     : BoolProperty          (name="By Face map",        description="Calculate id map based on face map membership", default=False)
    selectmaterial: BoolProperty(
        name="Only material id",
        description="Calculate id map only for faces with a material id (disabled when no materials present on mesh)",
        default=False,
    )
    selectid: IntProperty(
        name="Material id", description="Material id (slot index)", default=0, min=0
    )
    usedisplaycolor: BoolProperty(
        name="Use display color",
        description="Use display color of assigned material",
        default=False,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh.
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and len(context.selected_objects) > 0
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
        )
        return p

    @staticmethod
    def length_variance(coords, indices):
        verts = coords[indices]
        center = verts.mean(axis=0)
        dirs = verts - center
        lengths = np.einsum("ij,ij->i", dirs, dirs)
        return np.var(lengths)

    def mapcolors(self, ob):
        if ob.type != "MESH":
            return

        mesh = ob.data

        if mesh.color_attributes.active_color_index < 0:
            mesh.color_attributes.new(name="Color", domain="CORNER", type="BYTE_COLOR")
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        # we work on a bmesh because edge traversal etc. is faster
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        # assign a unique value to related groups of faces

        seed(self.seed)
        # the initial weightmap is empty
        weightmap = {}
        selected = self.selected
        selectmaterial = self.selectmaterial
        selectid = self.selectid

        if self.method == "MATERIALID":
            weightmap = {
                f.index: f.material_index if (not selected or f.select) else -1
                for f in bm.faces
            }
            if self.usedisplaycolor:
                nmats = len(ob.material_slots)
                # diffuse_color is a Color (3 components) whereas a vertex color layer has color attributes
                # which are not colors but 4 element vectors
                # also note that an object can have slots defined without a material assigned to it
                indexed_colors = {
                    w: (list(ob.material_slots[w].material.diffuse_color))
                    if (w < nmats and w >= 0 and ob.material_slots[w].material is not None)
                    else self.basecolor[:]
                    for w in set(weightmap.values())
                }
            else:
                indexed_colors = {
                    w: [random(), random(), random(), 1.0]
                    for w in set(weightmap.values())
                }  # assign a random color for each unique weight
            indexed_colors[-1] = list(
                self.basecolor[:]
            )  # but the default color for non selected faces is defined by the user
            # Optional: reduce number of distinct output colors
            if self.number_of_ids:
                ncolors = max(1, int(self.colors))
                # limit by available non-default weights
                ncolors = min(ncolors, len([w for w in set(weightmap.values()) if w != -1]))
                try:
                    palette = color_set(ncolors)
                except Exception:
                    palette = np.ones((ncolors, 4), dtype=np.float32)
                    for i in range(ncolors):
                        h = i / max(1, ncolors)
                        palette[i, :3] = [h, 1.0 - h, (0.5 + h) % 1.0]
                remap_keys = sorted([w for w in indexed_colors.keys() if w != -1])
                for i, w in enumerate(remap_keys):
                    c = palette[i % ncolors]
                    indexed_colors[w][:3] = [float(c[0]), float(c[1]), float(c[2])]
        # facemaps are currently unsupported in 4.0, see https://projects.blender.org/blender/blender/issues/105317
        # elif self.method == "FACEMAPID":
        #     fm = bm.faces.layers.face_map.verify()
        #     weightmap = {
        #         f.index: f[fm] if (not selected or f.select) else -1 for f in bm.faces
        #     }
        #     indexed_colors = {
        #         w: [random(), random(), random(), 1.0] for w in set(weightmap.values())
        #     }  # assign a random color for each unique weight
        #     indexed_colors[-1] = list(
        #         self.basecolor[:]
        #     )  # but the default color for non selected faces is defined by the user
        else:  # "HEURISTIC"
            # map this only once
            similarity = (self.similarity - 0.5) * 2.0  # map [0,1] -> [-1,1]

            # check if we have a bevel layer
            # bevel_layer = bm.edges.layers.bevel_weight.active
            bevel_layer = None
            if 'bevel_weight_edge' in bm.edges.layers.float.keys():
                bevel_layer = bm.edges.layers.float['bevel_weight_edge']
            bevel_weight = self.bevel_weight if bevel_layer else 0
            
            # check if we have a crease layer
            #crease_layer = bm.edges.layers.crease.active
            crease_layer = None
            if 'crease_edge' in bm.edges.layers.float.keys():
                crease_layer = bm.edges.layers.float['crease_edge']
            crease_weight = self.crease_weight if crease_layer else 0

            # assign a random value for each group of related faces
            faces = set(
                f.index
                for f in bm.faces
                if (f.select or not selected)
                and (f.material_index == selectid or not selectmaterial)
            )
            face = faces.pop() if len(faces) else None

            # sets are based on the indices of faces because the address of faces changes with every invocation of the operator because it creates a new bmesh.
            while face is not None:
                # a new related region
                weightmap[face] = random()
                todo = set()
                todo.add(face)

                aw = self.area_weight
                while len(todo):
                    f = todo.pop()
                    ff = bm.faces[f]
                    weightmap[f] = weightmap[face]
                    for edge in ff.edges:
                        # uv-seams, edges marked as sharp and/or strong bevel and crease weights might delimit areas
                        if self.seams and edge.seam:
                            continue
                        if self.sharp and not edge.smooth:
                            continue
                        if crease_weight > 0 and edge[crease_layer] > crease_weight:
                            continue
                        if bevel_weight > 0 and edge[bevel_layer] > bevel_weight:
                            continue
                        for other_face in edge.link_faces:
                            if (
                                selected and not other_face.select
                            ):  # skip not selected faces if requested
                                continue
                            if (
                                selectmaterial
                                and not other_face.material_index == selectid
                            ):  # skip faces with the wrong material id
                                continue
                            ofi = other_face.index
                            if (ofi in todo) or (
                                ofi in weightmap
                            ):  # face already visited or already on our todo list
                                continue
                            elif (
                                self.smooth and ff.smooth != other_face.smooth
                            ):  # faces with different shading attributes
                                continue
                            # two faces are considered similar if their face normals point more or less in the same direction
                            elif aw > 0:
                                aq = ff.calc_area() / other_face.calc_area()
                                if aq > 1:
                                    aq = (
                                        1 / aq
                                    )  # aq always <= 1, if aq == 1 areas are identical
                                nw = ff.normal.dot(other_face.normal)
                                if nw * (1 - aw) + nw * aq * aw > similarity:
                                    faces.remove(ofi)
                                    todo.add(ofi)
                            elif ff.normal.dot(other_face.normal) > similarity:
                                faces.remove(ofi)
                                todo.add(ofi)

                # next region
                face = faces.pop() if len(faces) else None

            # not all faces may have been assigned a value if we have restricted assignment to selected faces
            if selected:
                weightmap.update({f.index: -1 for f in bm.faces if not f.select})
            if selectmaterial:
                weightmap.update(
                    {f.index: -1 for f in bm.faces if f.material_index != selectid}
                )

            # the next, optional, step is to merge groups of adjacent faces
            # that have each been assigned distinct colors, for example because
            # they are part of a strip of faces with sharp angles that separate
            # two flat areas

            if self.merge:
                # first, collect all isolated faces
                isolated_faces = set()
                for face in bm.faces:
                    connected = False
                    for edge in face.edges:
                        for other_face in edge.link_faces:
                            # an isolated face has no neighbour with the same color as itself
                            # or equivalently a connected face has at least one neighbour with a color equal to its own
                            if (
                                other_face.index != face.index
                                and weightmap[other_face.index] == weightmap[face.index]
                            ):
                                connected = True
                                break
                        if connected:
                            break
                    if not connected:
                        isolated_faces.add(face.index)

                # then coalesce connected regions within the isolated faces if possible
                face = isolated_faces.pop() if len(isolated_faces) else None
                while face is not None:
                    weightmap[face] = random()
                    todo = set()
                    todo.add(face)
                    while len(todo):
                        f = todo.pop()
                        ff = bm.faces[f]
                        weightmap[f] = weightmap[face]
                        for edge in ff.edges:
                            for other_face in edge.link_faces:
                                if (
                                    (face != other_face.index)
                                    and (other_face.index in isolated_faces)
                                    and (other_face.index not in todo)
                                ):
                                    isolated_faces.remove(other_face.index)
                                    todo.add(other_face.index)
                    face = isolated_faces.pop() if len(isolated_faces) else None

            if self.match:
                # create sets of faces with the same weight (= color)
                face_sets = dd(set)
                for f, w in weightmap.items():
                    face_sets[w].add(f)
                remap = {}
                # assign the same weight to all sets of the same size
                if self.matchvar > 0:
                    coords = np.empty(len(mesh.vertices) * 3)
                    mesh.vertices.foreach_get("co", coords)
                    coords.shape = -1, 3
                    variance = {}
                    for w, fs in face_sets.items():
                        indices = {v.index for f in fs for v in bm.faces[f].verts}
                        variance[w] = self.length_variance(coords, list(indices))
                    maxvar = max(variance.values())

                    # determine the set of all available sets with the same approximate variance
                    nfaces = set()
                    setvars = {}
                    for w, fs in face_sets.items():
                        setvar = (
                            round(variance[w] / maxvar, self.matchvar)
                            if maxvar > 1e-7
                            else 0.0
                        )
                        setvars[w] = setvar
                        nfaces.add((len(fs), setvar))

                    for n, setvar in sorted(nfaces):
                        first = None
                        for w, fs in face_sets.items():
                            if setvar == setvars[w] and len(fs) == n:
                                if first is None:
                                    first = w
                                remap[w] = first

                    # print('unique set lengths  ', len({len(fs) for fs in face_sets.values()}))
                    # print('unique set variances', len(nfaces))
                    # print('initial weights     ', len(face_sets))
                else:
                    # determine the set of all available set sizes
                    nfaces = set()
                    for w, fs in face_sets.items():
                        nfaces.add(len(fs))
                    # merge them unconditionally if they have the same number of faces
                    for n in sorted(nfaces):
                        first = None
                        for w, fs in face_sets.items():
                            if len(fs) == n:
                                if first is None:
                                    first = w
                                remap[w] = first
                # do the actual remapping of weights
                for f, w in weightmap.items():
                    if w in remap:
                        weightmap[f] = remap[w]

                # print('final weights       ', len(set(weightmap.values())))

            # because we might have more than 256 distinct but related regions, and we don't want overlapping colors, we have to make our
            # randomly created colors more distinct.
            indexed_colors = {
                w: [random(), random(), random(), 1.0] for w in set(weightmap.values())
            }  # assign a random color for each unique weight
            indexed_colors[-1] = list(
                self.basecolor[:]
            )  # but the default color for non selected faces is defined by the user
            # Optional: reduce number of distinct output colors
            if self.number_of_ids:
                ncolors = max(1, int(self.colors))
                # limit by available non-default weights
                ncolors = min(ncolors, len([w for w in set(weightmap.values()) if w != -1]))
                try:
                    palette = color_set(ncolors)
                except Exception:
                    palette = np.ones((ncolors, 4), dtype=np.float32)
                    for i in range(ncolors):
                        h = i / max(1, ncolors)
                        palette[i, :3] = [h, 1.0 - h, (0.5 + h) % 1.0]
                remap_keys = sorted([w for w in indexed_colors.keys() if w != -1], key=lambda x: str(x))
                for i, w in enumerate(remap_keys):
                    c = palette[i % ncolors]
                    indexed_colors[w][:3] = [float(c[0]), float(c[1]), float(c[2])]

        for f, weight in weightmap.items():
            color = indexed_colors[weight]
            # we assume that loops in an unaltered bmesh are the same as in the original mesh
            # so that we can use the indices in a vertex color layer that is associated with the original
            for loop in bm.faces[f].loops:
                vertex_colors[
                    loop.index
                ].color = (
                    copy(color)  # can be a list or a tuple and tuple hasn't a copy methos but the copy function can handle that
                )  # colors are wrapped so make sure each gets its unique copy

        bm.free()  # we created it, so we are responsible for freeing it again

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        # TODO only operate on mesh objects
        if bpy.context.preferences.addons[__name__].preferences.allselected:
            for ob in context.selected_objects:
                self.mapcolors(ob)
        else:
            self.mapcolors(context.active_object)

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")

        return {"FINISHED"}

    def draw(self, context):
        nmats = len(context.active_object.material_slots)
        # face maps are currently unsupported in 4.0, see https://projects.blender.org/blender/blender/issues/105317
        nmaps = 0  # len(context.active_object.face_maps)
        layout = self.layout
        layout.label(
            text="Boundaries"
        )  # should we hide or disable this options if the appropriate layer is not present?
        row = layout.row()
        row.prop(self, "method")
        if self.method == "HEURISTIC":
            layout.prop(self, "seams")
            layout.prop(self, "sharp")
            layout.prop(self, "bevel_weight")
            layout.prop(self, "crease_weight")
            layout.prop(self, "smooth")
            layout.prop(self, "merge")
            row = layout.row()
            row.prop(self, "match")
            row.prop(self, "matchvar")

            layout.label(text="Options")
            row = layout.row()
            row.prop(self, "selectmaterial")
            if self.selectmaterial:
                row.prop(self, "selectid")
            row.active = nmats > 0
            row = layout.row()
            row.prop(self, "similarity")
            row.prop(self, "area_weight")
        elif self.method == "MATERIALID":
            if nmats > 0:
                layout.prop(self, "usedisplaycolor")
            else:
                layout.label(text="No materials present on this object")
        else:  # FACEMAPID
            layout.label(text="Face maps are no longer supported in 4.x")
            layout.label(text="See: https://projects.blender.org/blender/blender/issues/105317")
            if nmaps < 1:
                layout.label(text="No face maps present on this object")
        layout.label(text="Restrict map to")
        row = layout.row()
        row.prop(self, "selected")
        if self.selected or self.method == "MATERIALID":
            row.prop(self, "basecolor")
        row = layout.row()
        row.prop(self, "number_of_ids")
        if self.number_of_ids:
            row.prop(self, "colors", slider=True)

        if self.method in {"HEURISTIC", "FACEMAPID"} or self.usedisplaycolor:
            layout.prop(self, "seed")


class VertexColorMerge(bpy.types.Operator):
    bl_idname = "paint.vertexcolormerge"
    bl_label = "Vertex Color Merge"
    bl_description = "Merge two vertex color layers"
    bl_options = {"REGISTER", "UNDO"}

    operation: EnumProperty(
        name="Mode",
        description="Merge mode",
        items=[("Add", "Add", "Add"), ("Multiply", "Multiply", "Multiply")],
    )
    vcol1: StringProperty(name="Layer 1", description="Vertex color layer 1")
    vcol2: StringProperty(name="Layer 2", description="Vertex color layer 2")

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has at least one vertex color layer
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and len(context.object.data.color_attributes)
        )
        return p

    def draw(self, context):
        mesh = context.object.data
        layout = self.layout
        layout.prop(self, "operation")
        layout.prop_search(self, "vcol1", mesh, "color_attributes")
        layout.prop_search(self, "vcol2", mesh, "color_attributes")

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        self.ob = context.active_object
        mesh = context.object.data

        # create new vertex color layer to hold the merged result
        bpy.ops.geometry.color_attribute_add(domain="CORNER", data_type="BYTE_COLOR")
        mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].name = "Col.Merged"
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        vcol1 = (
            mesh.vertex_colors[self.vcol1].data
            if self.vcol1 in mesh.vertex_colors
            else None
        )
        vcol2 = (
            mesh.vertex_colors[self.vcol2].data
            if self.vcol2 in mesh.vertex_colors
            else None
        )

        default = (0, 0, 0, 1)
        if self.operation == "Add":
            for loop in mesh.loops:
                vi = loop.vertex_index
                c1 = (
                    default[:] if vcol1 is None else vcol1[loop.index].color
                )  # no need to copt default here because we always create a new color to assign
                c2 = default[:] if vcol2 is None else vcol2[loop.index].color
                # print(c1[0:3], c2[0:3], (c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]))
                vertex_colors[loop.index].color[:3] = [
                    c1[0] + c2[0],
                    c1[1] + c2[1],
                    c1[2] + c2[2],
                ]
        elif self.operation == "Multiply":
            for loop in mesh.loops:
                vi = loop.vertex_index
                c1 = (
                    default[:] if vcol1 is None else vcol1[loop.index].color
                )  # no need to copt default here because we always create a new color to assign
                c2 = default[:] if vcol2 is None else vcol2[loop.index].color
                vertex_colors[loop.index].color[:3] = [
                    c1[0] * c2[0],
                    c1[1] * c2[1],
                    c1[2] * c2[2],
                ]

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80 scene.update()

        return {"FINISHED"}


class VertexColorFromSelected(bpy.types.Operator):
    bl_idname = "paint.vertexcolorfromselected"
    bl_label = "Vertex Color From Selected"
    bl_description = "Apply a color to faces that are selected in edit mode"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has at least one vertex color layer
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and len(context.object.data.color_attributes)
        )
        return p

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        self.ob = context.active_object
        mesh = context.object.data

        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        color = context.tool_settings.vertex_paint.brush.color

        indices = set()
        indices.update(
            loop
            for f in mesh.polygons
            for loop in range(f.loop_start, f.loop_start + f.loop_total)
            if f.select
        )
        for i in indices:
            vertex_colors[i].color[:3] = color.copy()

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80  scene.update()

        return {"FINISHED"}


def copy_vertex_colors(me_from, vc_from, me_to, vc_to, offset):
    bm_from = bmesh.new()
    bm_from.from_mesh(me_from)
    bm_to = bmesh.new()
    bm_to.from_mesh(me_to)

    bm_from.faces.ensure_lookup_table()
    bvh_from = BVHTree.FromBMesh(bm_from)
    for face in bm_to.faces:
        direction = face.normal
        src = face.calc_center_median() - offset * direction

        location, normal, face_index, distance = bvh_from.ray_cast(src, direction)
        locationb, normalb, face_indexb, distanceb = bvh_from.ray_cast(src, -direction)

        if location is None:
            location, normal, face_index, distance = (
                locationb,
                normalb,
                face_indexb,
                distanceb,
            )
        elif locationb is None:
            pass
        else:
            if distanceb < distance:
                location, normal, face_index, distance = (
                    locationb,
                    normalb,
                    face_indexb,
                    distanceb,
                )

        if location:
            from_face = bm_from.faces[face_index]
            color = vc_from[from_face.loops[0].index].color
        else:
            color = [0, 0, 0]
        for loop in face.loops:
            vc_to[loop.index].color[:3] = color[:3]

    bm_from.free()
    bm_to.free()


class VertexColorCopy(bpy.types.Operator):
    bl_idname = "mesh.vertexcolorcopy"
    bl_label = "Vertex Color Copy"
    bl_description = "Copy vertex color layer from another mesh"
    bl_options = {"REGISTER", "UNDO"}

    mesh_lo_name: StringProperty(
        name="Source mesh", description="Source mesh to copy vertex colors from"
    )
    offset: FloatProperty(
        name="Offset", description="Offset for ray cast", default=0.001, min=0, max=1
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has at least one vertex color layer
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and len(context.object.data.color_attributes)
        )
        return p

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        self.ob = context.active_object
        mesh_hi = context.object.data
        vertex_colors_hi = mesh_hi.color_attributes[
            mesh_hi.color_attributes.active_color_index
        ].data
        if self.mesh_lo_name != "":
            mesh_lo = bpy.data.meshes[self.mesh_lo_name]
            try:
                vertex_colors_lo = mesh_lo.color_attributes[
                    mesh_lo.color_attributes.active_color_index
                ].data
                copy_vertex_colors(
                    mesh_lo, vertex_colors_lo, mesh_hi, vertex_colors_hi, self.offset
                )
            except e:
                print(e)

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80 scene.update()

        return {"FINISHED"}

    def draw(self, context):
        mesh = context.object.data
        layout = self.layout
        layout.prop_search(self, "mesh_lo_name", bpy.data, "meshes")
        layout.prop(self, "offset")


from mathutils.bvhtree import BVHTree


def pick_color(self, context, event):
    """Return Color,face index tuple for point under pointer"""
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    ray_target = ray_origin + view_vector

    bvh = self.bvh

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv @ ray_origin
        ray_target_obj = matrix_inv @ ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        # cast the ray
        location, normal, face_index, distance = bvh.ray_cast(
            ray_origin_obj, ray_direction_obj
        )

        if location is not None:
            return location, normal, face_index
        else:
            return None, None, None

    obj = context.active_object

    if obj.type == "MESH":
        hit, normal, face_index = obj_ray_cast(obj, obj.matrix_world.copy())
        if hit is not None:
            vertex_colors = (
                obj.data.color_attributes.active_color.data
                # < 3.2 obj.data.vertex_colors.active.data
            )  # shoudl we check for None?
            loop_start = obj.data.polygons[face_index].loop_start
            return (
                vertex_colors[loop_start].color[:],
                face_index,
            )  # we are not actually interested in the loop only its index so we use it directly to index the vertex color layer
            # also, we copy the color because colors are wrapped values!
    # buf = bgl.Buffer(
    #     bgl.GL_FLOAT, [1, 3]
    # )  # Note that if we define this buffer on a global level it gets corrupted
    # bgl.glReadPixels(event.mouse_x, event.mouse_y, 1, 1, bgl.GL_RGB, bgl.GL_FLOAT, buf)
    # rgb = buf[0]
    fb = gpu.state.active_framebuffer_get()
    rgb = fb.read_color(event.mouse_x, event.mouse_y, 1, 1, 3,0,'FLOAT')[0][0]
    return (rgb[0], rgb[1], rgb[2], 1.0), None


def set_color(bm, bvh, context, event, color, restrict, restriction_color):
    """
    set vertex colors of face under mouse pointer to color. If restrict is true only apply a new color if the original color of the face is equal to restriction_color

    all colors are assumed to be 4-vectors
    """
    color = [color.r, color.g, color.b, 1.0]
    if restriction_color is None:
        restrict = False
    # get the context arguments
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    radius = (
        context.tool_settings.unified_paint_settings.size
    )  # context.tool_settings.vertex_paint.brush.size
    radius *= radius

    # get the ray from the viewport and mouse
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    ray_target = ray_origin + view_vector

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""

        # get the ray relative to the object
        matrix_inv = matrix.inverted()
        ray_origin_obj = matrix_inv @ ray_origin
        ray_target_obj = matrix_inv @ ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        # cast the ray
        location, normal, face_index, distance = bvh.ray_cast(
            ray_origin_obj, ray_direction_obj
        )

        if location is not None:
            return location, normal, face_index
        else:
            return None, None, None

    obj = context.active_object

    original_color = None
    if obj.type == "MESH":
        hit, normal, face_index = obj_ray_cast(obj, obj.matrix_world.copy())
        if hit is not None:
            original_color = None
            m_inv = obj.matrix_world.inverted()
            vertex_colors = (
                obj.data.color_attributes.active_color.data
                # < 3.2 obj.data.vertex_colors.active.data
            )  # shoudl we check for None?
            for f in bm.faces:
                f.tag = False
            faces = set()
            faces.add(bm.faces[face_index])
            faces_to_color = set()
            while len(faces):
                face = faces.pop()
                faces_to_color.add(face)
                face.tag = True
                for edge in face.edges:
                    for f in edge.link_faces:  # check neighboring faces
                        if (
                            not f.tag
                        ):  # exclude center face and any faces already visited
                            for v in f.verts:
                                if (
                                    v in face.verts
                                ):  # we are only interested in verts not shared with the central face
                                    continue
                                world_co = obj.matrix_world @ v.co
                                screen_co = view3d_utils.location_3d_to_region_2d(
                                    region, rv3d, world_co, default=None
                                )
                                if screen_co is None:
                                    break
                                x = screen_co[0] - coord[0]
                                y = screen_co[1] - coord[1]
                                if (
                                    x * x + y * y < radius
                                ):  # vert in range of cursor radius
                                    view_vector = view3d_utils.region_2d_to_vector_3d(
                                        region, rv3d, screen_co
                                    )
                                    ray_origin = view3d_utils.region_2d_to_origin_3d(
                                        region, rv3d, screen_co
                                    )
                                    ray_target = ray_origin + view_vector
                                    ray_origin_obj = m_inv @ ray_origin
                                    ray_target_obj = m_inv @ ray_target
                                    ray_direction_obj = ray_target_obj - ray_origin_obj
                                    (
                                        location,
                                        normal,
                                        face_index,
                                        distance,
                                    ) = bvh.ray_cast(ray_origin_obj, ray_direction_obj)
                                    if location is not None and (face_index == f.index):
                                        faces.add(
                                            f
                                        )  # just one qualifying vertex is enough to include the face
                                        break
                                    elif (
                                        location is not None
                                    ):  # ray cast may assign a random face if they share a vertex
                                        fhits = bm.faces[face_index]
                                        fndone = False
                                        for (
                                            vn
                                        ) in (
                                            fhits.verts
                                        ):  # we check if one of the vertices of the face is indeed close to the location of the vert we checked
                                            if (vn.co - location).length < 0.001:
                                                faces.add(f)
                                                fndone = True
                                                break
                                        if fndone:
                                            break
                        f.tag = (
                            True  # even if we didn add is this face we mark it as seen
                        )
            for f in faces_to_color:
                # print(vertex_colors[f.loops[0].index].color, restriction_color)
                if (
                    not restrict
                    or vertex_colors[f.loops[0].index].color[:] == restriction_color
                ):
                    for loop in f.loops:
                        vertex_colors[loop.index].color = color[
                            :
                        ]  # note that color should be a unique vector (not wrapped)


def flatten(bm, vertex_colors):
    """apply uniform colors to all vertices of a face. loop[0] is arbitrarily chosen as the color to set"""
    for f in bm.faces:
        color = vertex_colors[f.loops[0].index].color[:3]
        for loop in f.loops:
            vertex_colors[loop.index].color[
                :3
            ] = color  # note that color should be a unique vector (not wrapped)


def set_region(bm, vertex_colors, match_color, face_index, color):
    """
    Set vertex colors of face with face_id to color and also any neighboring face that has the match_color.

    matchcolor is a 4-vector, color a Color (a 3-vector)
    """
    for f in bm.faces:
        f.tag = False
    faces = set()
    faces.add(
        bm.faces[face_index]
    )  # first one we add automatically matches the match_color
    while len(faces):
        face = faces.pop()
        face.tag = True
        for loop in face.loops:
            vertex_colors[loop.index].color[:3] = color[
                :3
            ]  # we are not actually interested in the loop only its index so we use it directly to index the vertex color layer
        for edge in face.edges:
            for f in edge.link_faces:
                if (
                    vertex_colors[f.loops[0].index].color[:3] == match_color[:3]
                ):  # assume all loops of a face have the same color
                    if not f.tag:
                        faces.add(f)


def resize_up(
    bm,
    vertex_colors,
    matchcolor,
    face_index,
    restrict_color,
    respect_smooth,
    respect_seam,
):
    """Enlarge the same colored region face_index belongs to."""
    # first find all faces in the color connected region (tag them)
    for f in bm.faces:
        f.tag = False
    faces = set()
    faces.add(
        bm.faces[face_index]
    )  # first one we add automatically matches the match_color
    while len(faces):
        face = faces.pop()
        face.tag = True
        for edge in face.edges:
            for f in edge.link_faces:
                if (
                    vertex_colors[f.loops[0].index].color[:3] == matchcolor[:3]
                ):  # assume all loops of a face have the same color
                    if not f.tag:
                        faces.add(f)
    # then check their neighbors
    if restrict_color is None:
        for f in bm.faces:
            if f.tag:
                for edge in f.edges:
                    if respect_smooth and not edge.smooth:
                        continue
                    if respect_seam and edge.seam:
                        continue
                    for fn in edge.link_faces:
                        if not fn.tag:  # exclude self
                            for loop in fn.loops:
                                vertex_colors[loop.index].color[:3] = matchcolor[:3]
    else:
        for f in bm.faces:
            if f.tag:
                for edge in f.edges:
                    if respect_smooth and not edge.smooth:
                        continue
                    if respect_seam and edge.seam:
                        continue
                    for fn in edge.link_faces:
                        if not fn.tag:
                            if (
                                vertex_colors[fn.loops[0].index].color[:3]
                                == restrict_color[:3]
                            ):  # exclude self and faces that don't have the brush color
                                for loop in fn.loops:
                                    vertex_colors[loop.index].color[:3] = matchcolor[:3]


def resize_down(
    bm,
    vertex_colors,
    matchcolor,
    face_index,
    restrict_color,
    respect_smooth,
    respect_seam,
):
    """Shrink the same colored region face_index belongs to."""
    # first find all faces in the color connected region (tag them)
    for f in bm.faces:
        f.tag = False
    faces = set()
    faces.add(
        bm.faces[face_index]
    )  # first one we add automatically matches the match_color
    while len(faces):
        face = faces.pop()
        face.tag = True
        for edge in face.edges:
            for f in edge.link_faces:
                if (
                    vertex_colors[f.loops[0].index].color[:3] == matchcolor[:3]
                ):  # assume all loops of a face have the same color
                    if not f.tag:
                        faces.add(f)
    # then check their neighbors
    if restrict_color is None:
        for f in bm.faces:
            if f.tag:
                done = False
                for edge in f.edges:
                    if respect_smooth and not edge.smooth:
                        continue
                    if respect_seam and edge.seam:
                        continue
                    for fn in edge.link_faces:
                        if not fn.tag:  # exclude self
                            color = vertex_colors[fn.loops[0].index].color[:3]
                            if color != matchcolor[:3]:
                                for loop in f.loops:
                                    vertex_colors[loop.index].color[:3] = color
                                done = True
                                break
                    if done:
                        break
    else:
        for f in bm.faces:
            if f.tag:
                done = False
                for edge in f.edges:
                    if respect_smooth and not edge.smooth:
                        continue
                    if respect_seam and edge.seam:
                        continue
                    for fn in edge.link_faces:
                        if not fn.tag:  # exclude self
                            color = vertex_colors[fn.loops[0].index].color[:3]
                            if color == restrict_color[:3]:
                                for loop in f.loops:
                                    vertex_colors[loop.index].color[:3] = color
                                done = True
                                break
                    if done:
                        break


def smooth(
    bm,
    vertex_colors,
    matchcolor,
    face_index,
    restrict_color,
    respect_smooth,
    respect_seam,
):
    # first find all faces in the color connected region (tag them)
    for f in bm.faces:
        f.tag = False
    faces = set()
    faces.add(
        bm.faces[face_index]
    )  # first one we add automatically matches the match_color
    while len(faces):
        face = faces.pop()
        face.tag = True
        for edge in face.edges:
            for f in edge.link_faces:
                if (
                    vertex_colors[f.loops[0].index].color[:3] == matchcolor[:3]
                ):  # assume all loops of a face have the same color
                    if not f.tag:
                        faces.add(f)
    change = {}
    nonmatch = [0, 0, 0]
    for face in bm.faces:
        faces, matches, non_matches = 0, 0, 0
        fc = vertex_colors[face.loops[0].index].color[:3]
        if (
            face.tag
        ):  # this face is part of the colored area, we are going to see if we will erode it (== five it a different color)
            for edge in face.edges:
                if respect_smooth and not edge.smooth:
                    continue
                if respect_seam and edge.seam:
                    continue
                for fn in edge.link_faces:
                    faces += 1
                    nfc = vertex_colors[fn.loops[0].index].color[:3]
                    if nfc == matchcolor[:3]:
                        matches += 1
                    else:
                        non_matches += 1
                        nonmatch = nfc
            if matches == 1:  # it only has 1 matching neighbor
                change[face] = restrict_color[:3] if restrict_color else nonmatch
        elif (restrict_color is None) or (
            fc == restrict_color[:3]
        ):  # this face is part of the restricted color or we consider any adjacent face if there is no color restriction
            tag = False
            for edge in face.edges:  # we check if this face is going to be dilated into
                if respect_smooth and not edge.smooth:
                    continue
                if respect_seam and edge.seam:
                    continue
                for fn in edge.link_faces:
                    faces += 1
                    nfc = vertex_colors[fn.loops[0].index].color[:3]
                    if nfc == matchcolor[:3]:
                        matches += 1
                    else:
                        non_matches += 1
                    if fn.tag:
                        tag = True
            if tag:  # we are next to the colored area
                if matches > 1:
                    change[face] = matchcolor[:3]
    for face, color in change.items():
        for loop in face.loops:
            vertex_colors[loop.index].color[:3] = color[:3]


def recolor(vertex_colors, match_color, color):
    """Replace the vertex colors that match a certain color."""
    for dataelement in vertex_colors:
        print(dataelement.color[:], match_color)
        if dataelement.color[:] == match_color[:]:
            dataelement.color[:3] = color


def unique_2d(
    a,
):  # Blender 2.79 does not inlude numpy 1.13 so we do not have an axis argument in np.unique, but see https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """
    Return a unique list of Mx3 from an input of Nx3.
    """
    b = a[np.lexsort(a.T)]
    return b[np.concatenate(([True], np.any(b[1:] != b[:-1], axis=1)))]


def unique_3d(
    obcolors,
):
    """
    Return a unique list of Mx4 from an input of dict[ob] -> Nx4.
    """
    N = sum(len(colors) for colors in obcolors.values())
    a = np.empty((N, 5), dtype=np.float32)
    offset = 0
    obmap = {}
    for nid, (ob, colors) in enumerate(obcolors.items()):
        L = len(colors)
        a[offset : offset + L, 1:] = colors[:, :]
        a[offset : offset + L, 0] = nid
        offset += L
        obmap[ob] = nid
    a.shape = -1, 5
    return np.unique(a, axis=0), obmap


def color_set(n):
    """
    Create a list of n colors that are as far apart as possible in the color cube.
    """
    n += 2  # to exclude pure black and pure white later
    N = 2
    if n >= 8:
        N = 1 + int(pow(n - 1, 1 / 3))
    colors = np.empty(N * N * N * 4, dtype=np.float32)
    colors.shape = N, N, N, 4
    M = N - 1
    for r in range(N):
        for g in range(N):
            for b in range(N):
                colors[r, g, b] = [r / M, g / M, b / M, 1.0]
    colors.shape = -1, 4
    np.random.shuffle(
        colors[1 : n - 1]
    )  # we exclude pure black and white to prevent confusion with non masked areas
    return colors[1 : n - 1]


# each line MUST have two tabs
paint_helptext = [
    "Left   mouse   paint",
    "   S   pick color",
    "   K   fill region",
    "Alt    K   fill same colored regions",
    "   W   smooth (ctrl respects seams)",
    "Alt    W   restricted smooth (ctrl respects seams)",
    "   P   paint selected faces",
    "   F   resize cursor (+mousewheel)",
    "Num    +   expand region (ctrl respects seams)",
    "Num    -   shrink region (ctrl respects seams)",
    "Num    /   flatten face colors",
    "Ctl    Z   undo",
    "Ctl    1-9 select from color list",
    "Ctl    0   roll color list up",
]

blf.size(0, 10)

top = 10 + 12 * len(paint_helptext)
_parts = [ht.split("\t") for ht in paint_helptext]
_parts = [(p + ["", "", ""])[:3] for p in _parts]
ph1 = [p[0] for p in _parts]
w1, _ = blf.dimensions(0, max(ph1, key=len) if ph1 else "")
ph2 = [p[1] for p in _parts]
w2, _ = blf.dimensions(0, max(ph2, key=len) if ph2 else "")
ph3 = [p[2] for p in _parts]
w3, _ = blf.dimensions(0, max(ph3, key=len) if ph3 else "")
left3 = w3 + 10
left2 = w2 + 4 + left3
left1 = w1 + 8 + left2

# no longer needed buf = bgl.Buffer(bgl.GL_INT, 4)  # linear array of 4 ints
showhelp = True

from gpu_extras import presets as gpupresets
import gpu
from gpu_extras.batch import batch_for_shader

shader = gpu.shader.from_builtin("UNIFORM_COLOR")
shader.bind()
shader.uniform_float("color", (1, 1, 1, 0.5))


def cursor_handler(self, context, shortcut):
    """Draw a circle around the mouse pointer with a radius equal the current brush size."""

    # 50% alpha, 1 pixel circle
    # deprecated: bgl.glEnable(bgl.GL_BLEND)

    width = gpu.state.line_width_get()
    gpu.state.line_width_set(1)
    r = (
        context.tool_settings.unified_paint_settings.size
    )  # context.tool_settings.vertex_paint.brush.size

    if (
        not shortcut
    ):  # if invoked from shortcut the vertex paint cursor stays on screen so we don't have to draw the circle ourselves
        gpupresets.draw_circle_2d(self.mouse_pos, (1.0, 1.0, 1.0, 0.5), r, segments=32)

    coords = []
    for x1, y1, x2, y2 in self.bracket:
        coords.append(
            (self.mouse_pos[0] + int(x1 * r), self.mouse_pos[1] + int(y1 * r), 0)
        )
        coords.append(
            (self.mouse_pos[0] + int(x2 * r), self.mouse_pos[1] + int(y2 * r), 0)
        )
    batch = batch_for_shader(shader, "LINES", {"pos": coords})
    batch.draw(shader)

    # help display
    if showhelp:
        buf = gpu.state.viewport_get()
        screenwidth = buf[2] - 10
        blf.size(0, 10)

        blf.color(0, *context.preferences.addons[__name__].preferences.helptextcolor)

        for no, line in enumerate(ph1):
            blf.position(0, screenwidth - left1, top - no * 12, 0)
            blf.draw(0, line)
        for no, line in enumerate(ph2):
            blf.position(0, screenwidth - left2, top - no * 12, 0)
            blf.draw(0, line)
        for no, line in enumerate(ph3):
            blf.position(0, screenwidth - left3, top - no * 12, 0)
            blf.draw(0, line)

    gpu.state.line_width_set(width)

    # restore opengl defaults
    # deprecated: bgl.glLineWidth(1)
    # deprecated: bgl.glDisable(bgl.GL_BLEND)


class VertexColorFacePaint(bpy.types.Operator):
    bl_idname = "paint.vertexcolorfacepaint"
    bl_label = "Face Paint"
    bl_description = "Face Paint mode"

    def _set_cursor(self, context, mode):
        if self.cursor is not None:
            context.window.cursor_modal_restore()
        self.cursor = None
        if mode == "PICK":
            self.cursor = mode
            context.window.cursor_modal_set("EYEDROPPER")
        elif mode == "PAINT":
            self.cursor = mode
            context.window.cursor_modal_set("CROSSHAIR")
        elif mode == "RESIZE":
            self.cursor = mode
            context.window.cursor_modal_set("SCROLL_XY")
        context.area.tag_redraw()
        
    def modal(self, context, event):
        global showhelp

        helptext = {
            None: "Click to paint, S: pick color, K: fill, H: toggle help, ESC/right mouse to exit",
            "PICK": "[Color pick]: release S to exit",
            "PAINT": "[Paint] Mousewheel to resize brush, Alt restrict painting to current area",
            "CURSORRESIZE": "[Resize brush] Mousewheel to resize brush, release F to exit",
        }

        context.area.header_text_set(helptext[self.mode])
        context.tool_settings.vertex_paint.show_brush = True  # <-- this doesn't work
        context.area.tag_redraw()

        self.mouse_pos = (event.mouse_region_x, event.mouse_region_y)

        if (
            event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE", "MIDDLEMOUSE"}
            and not event.alt
            and not self.mode in ("PAINT", "CURSORRESIZE")
        ):
            # allow navigation but we steal middlemouse actions when modified with alt or when in cursor resize mode
            return {"PASS_THROUGH"}
        elif event.type in {
            "NUMPAD_PERIOD",
            "TAB",
            "HOME",
            "NUMPAD_2",
            "NUMPAD_4",
            "NUMPAD_6",
            "NUMPAD_8",
            "NUMPAD_1",
            "NUMPAD_3",
            "NUMPAD_7",
            "NUMPAD_9",
            "NUMPAD_5",
        }:
            # allow some keyboard navigation. Note that passing on TAB will switch to edit mode and back, but if someone edits the mesh a crash might be the result? apperently not because the modal mode gets canceled if we come back from edit mode
            return {"PASS_THROUGH"}
        elif (
            event.type in {"RIGHTMOUSE", "ESC"} and not event.shift
        ):  # we allow the customary escape and right mouselick to end the modal operation but steal shift-rightclick
            self.bm.free()
            self._set_cursor(context, None)
            if self._handle:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            context.area.header_text_set(None)
            return {"CANCELLED"}

        # print(event.type, event.value, event.ascii, event.shift, event.ctrl, event.alt, self.mode, self.modify)
        try:  # just to make sure we can free resources, reset handler etc. we catch *anything* unexpected
            numkeys = {
                "ONE": 1,
                "TWO": 2,
                "THREE": 3,
                "FOUR": 4,
                "FIVE": 5,
                "SIX": 6,
                "SEVEN": 7,
                "EIGHT": 8,
                "NINE": 9,
            }
            if self.mode is None:
                if event.type == "H" and event.value == "RELEASE":
                    showhelp = not showhelp
                elif event.type == "Z" and event.value == "RELEASE":
                    self.undo()
                    self.mesh.update()

                elif event.type == "S" and event.value == "PRESS":
                    self.mode = "PICK"
                    self._set_cursor(context, self.mode)
                    color, face_index = pick_color(self, context, event)
                    if color is not None:  # always True
                        context.tool_settings.vertex_paint.brush.color = color[
                            :3
                        ]  # brush color does not have alpha

                elif event.type == "K" and event.value == "RELEASE":
                    matchcolor, face_index = pick_color(self, context, event)
                    if face_index is not None:
                        self.set_undo()
                        if event.alt:
                            recolor(
                                self.vertex_colors,
                                matchcolor,
                                context.tool_settings.vertex_paint.brush.color,
                            )
                        else:
                            set_region(
                                self.bm,
                                self.vertex_colors,
                                matchcolor,
                                face_index,
                                context.tool_settings.vertex_paint.brush.color,
                            )

                elif event.type == "W" and event.value == "RELEASE":
                    matchcolor, face_index = pick_color(self, context, event)
                    if face_index is not None:
                        self.set_undo()
                        smooth(
                            self.bm,
                            self.vertex_colors,
                            matchcolor,
                            face_index,
                            context.tool_settings.vertex_paint.brush.color
                            if event.alt
                            else None,
                            event.ctrl,
                            event.ctrl,
                        )

                elif event.type == "F" and event.value == "PRESS":
                    self.mode = "CURSORRESIZE"
                    self._set_cursor(context, self.mode)

                elif event.type == "LEFTMOUSE" and event.value == "PRESS":
                    self.mode = "PAINT"
                    self.set_undo()
                    self.original_color, face_index = pick_color(
                        self, context, event
                    )  # this might be set to None if the cursor is not on the mesh
                    set_color(
                        self.bm,
                        self.bvh,
                        context,
                        event,
                        context.tool_settings.vertex_paint.brush.color,
                        event.alt,
                        self.original_color,
                    )
                    self._set_cursor(context, self.mode)

                elif event.type == "P" and event.value == "RELEASE":
                    self.set_undo()
                    color = context.tool_settings.vertex_paint.brush.color
                    indices = set()
                    indices.update(
                        loop
                        for f in self.mesh.polygons
                        for loop in range(f.loop_start, f.loop_start + f.loop_total)
                        if f.select
                    )
                    for i in indices:
                        self.vertex_colors[i].color[:3] = color.copy()
                elif event.type in numkeys and event.ctrl and event.value == "RELEASE":
                    bpy.ops.scene.colormap_set_active(index=numkeys[event.type] - 1)
                elif event.type == "ZERO" and event.ctrl and event.value == "RELEASE":
                    bpy.ops.scene.colormap_roll_up()
                # checking for an ascii == '+' might be done by storing the press event and ignoring anuthing until a release event comes along and then executing the code and clearing the pressed status
                elif (
                    event.type == "NUMPAD_PLUS" and event.value == "RELEASE"
                ):  # note that we cannot check on event.ascii == '+' because ascii values are only set on press!
                    matchcolor, face_index = pick_color(self, context, event)
                    if face_index is not None:
                        self.set_undo()
                        resize_up(
                            self.bm,
                            self.vertex_colors,
                            matchcolor,
                            face_index,
                            context.tool_settings.vertex_paint.brush.color
                            if event.alt
                            else None,
                            event.ctrl,
                            event.ctrl,
                        )
                elif event.type == "NUMPAD_MINUS" and event.value == "RELEASE":
                    matchcolor, face_index = pick_color(self, context, event)
                    if face_index is not None:
                        self.set_undo()
                        resize_down(
                            self.bm,
                            self.vertex_colors,
                            matchcolor,
                            face_index,
                            context.tool_settings.vertex_paint.brush.color
                            if event.alt
                            else None,
                            event.ctrl,
                            event.ctrl,
                        )
                elif event.type == "NUMPAD_SLASH" and event.value == "RELEASE":
                    self.set_undo()
                    flatten(self.bm, self.vertex_colors)

            elif self.mode == "PICK":  # S was pressed
                if event.type == "S" and event.value == "RELEASE":
                    self.mode = None
                    self._set_cursor(context, self.mode)
                elif event.type == "MOUSEMOVE" and event.value == "PRESS":
                    color, face_index = pick_color(self, context, event)
                    if color is not None:  # always True
                        context.tool_settings.vertex_paint.brush.color = color[:3]

            elif self.mode == "PAINT":  # Mouse was pressed
                if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                    self.mode = None
                    self._set_cursor(context, self.mode)
                elif (
                    event.type == "MOUSEMOVE"
                ):  # and event.value == 'PRESS':  # moving while pressed/non pressed
                    set_color(
                        self.bm,
                        self.bvh,
                        context,
                        event,
                        context.tool_settings.vertex_paint.brush.color,
                        event.alt,
                        self.original_color,
                    )
                elif event.type == "WHEELUPMOUSE":
                    context.tool_settings.unified_paint_settings.size += 1
                elif event.type == "WHEELDOWNMOUSE":
                    if context.tool_settings.unified_paint_settings.size > 1:
                        context.tool_settings.unified_paint_settings.size -= 1

            elif self.mode == "CURSORRESIZE":  # Mouse was pressed
                if event.type == "F" and event.value == "RELEASE":
                    self.mode = None
                    self._set_cursor(context, self.mode)
                elif event.type == "WHEELUPMOUSE":
                    context.tool_settings.unified_paint_settings.size += 1
                elif event.type == "WHEELDOWNMOUSE":
                    if context.tool_settings.unified_paint_settings.size > 1:
                        context.tool_settings.unified_paint_settings.size -= 1
        except:
            self.bm.free()
            self._set_cursor(context, None)
            if self._handle:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            context.area.header_text_set("")
            traceback.print_exc()  # print the exception plus stacktrace
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def set_undo(self):
        UNDODEPTH = bpy.context.preferences.addons[__name__].preferences.undodepth
        self.undo_stacklevel += 1
        if self.undo_stacklevel >= UNDODEPTH:
            self.undo_stacklevel -= 1
            for i in range(UNDODEPTH - 1):
                self.undo_stack[i] = self.undo_stack[i + 1]
        v = self.undo_stack[self.undo_stacklevel]
        p = v.reshape(-1)
        self.vertex_colors.foreach_get("color", p)

    def undo(self):
        if self.undo_stacklevel < 0:
            return
        v = self.undo_stack[self.undo_stacklevel]
        p = v.reshape(-1)
        self.vertex_colors.foreach_set("color", p)
        self.undo_stacklevel -= 1

    def invoke(self, context, event):
        self.mouse_pos = (
            event.mouse_region_x,
            event.mouse_region_y,
        )  # need this for our cursor draw handler

        UNDODEPTH = bpy.context.preferences.addons[__name__].preferences.undodepth

        if context.space_data.type == "VIEW_3D":
            # no longer present in 2.80 context.scene.update()

            obj = context.active_object
            self.mesh = obj.data
            # select the active vertex color layer or create one if it does not exist yet
            # this may happen if we stay in vertex paint mode but remove vcolor layers before starting face paint

            if self.mesh.color_attributes.active_color_index < 0:
                bpy.ops.geometry.color_attribute_add(
                    domain="CORNER", data_type="BYTE_COLOR"
                )
            self.vertex_colors = self.mesh.color_attributes.active_color.data

            self.nloops = len(self.vertex_colors)
            self.undo_stack = np.empty(UNDODEPTH * self.nloops * 4, dtype=np.float32)
            self.undo_stack.shape = UNDODEPTH, self.nloops, 4
            self.undo_stacklevel = -1

            # we work on a bmesh because edge traversal etc. is faster
            self.bm = bmesh.new()
            self.bm.from_mesh(self.mesh)
            self.bm.faces.ensure_lookup_table()
            self.bvh = BVHTree.FromBMesh(self.bm)

            self.modify = False
            self.mode = None
            self.cursor = None
            self.restrict_size = False

            # add a draw handler that draws a circle
            args = (self, context, (event.type == "P"))

            segments = 32
            self.circle = []
            for s in range(segments):
                d = 6.183 * s / segments
                self.circle.append((sin(d), cos(d)))
            self.circle.append(self.circle[0])
            self.bracket = [
                (-1, -0.5, -1, 0.5),
                (1, -0.5, 1, 0.5),
                (-0.5, 1, 0.5, 1),
                (-0.5, -1, 0.5, -1),
            ]

            if (
                context.area.type == "VIEW_3D"
            ):  # only add handler if not invoked via shortcut becuase then the regular circle is still there
                self._handle = bpy.types.SpaceView3D.draw_handler_add(
                    cursor_handler, args, "WINDOW", "POST_PIXEL"
                )
            else:
                self._handle = None
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()

            return {"RUNNING_MODAL"}
        else:
            self.report({"WARNING"}, "Active space must be a View3d")
            return {"CANCELLED"}


class VGroupsToVertexColor(bpy.types.Operator):
    bl_idname = "mesh.vgroupstovertexcolor"
    bl_label = "VGroups To Vertex Color"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Assign vertex colors based on vertex group membership"

    cutoff_weight: FloatProperty(
        name="Weight threshold",
        description="Minimum weight considered for vertex group membership",
        default=0,
        min=0,
        max=1,
    )
    basecolor: FloatVectorProperty(
        name="Base color",
        description="Color assigned to faces not in a vertex group",
        subtype="COLOR",
        default=Vector((0, 0, 0)),
    )
    replace: BoolProperty(
        name="Replace",
        description="Assign Base Color if vertex not a member of a vertex group",
        default=False,
    )
    fullface: BoolProperty(
        name="Full face", description="Give faces uniform colors", default=True
    )
    fullfaceall: BoolProperty(
        name="All", description="Only faces with all verts selected", default=True
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed (different values give different colors)",
        default=0,
        min=0,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has vertex groups.
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and len(context.object.vertex_groups) > 0
        )
        return p

    def draw(self, context):
        col = self.layout
        col.prop(self, "cutoff_weight")
        col.prop(self, "replace")
        if self.replace:
            col.prop(self, "basecolor")
        row = col.row()
        row.prop(self, "fullface")
        if self.fullface:
            row.prop(self, "fullfaceall")
        col.prop(self, "seed")

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        self.ob = context.active_object
        mesh = context.object.data

        seed(self.seed)

        # select the active vertex color layer or create one if it does not exist yet
        if mesh.color_attributes.active_color_index < 0:
            bpy.ops.geometry.color_attribute_add(
                domain="CORNER", data_type="BYTE_COLOR"
            )
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        # loop over all vertex groups
        if self.replace:
            for loop in mesh.loops:
                vertex_colors[loop.index].color[:3] = self.basecolor.copy()
        t = self.cutoff_weight  # promote to local variable for performance
        for vertex_group in self.ob.vertex_groups:
            color = [random(), random(), random()]
            if self.fullface:
                for p in mesh.polygons:
                    weight = None
                    if self.fullfaceall:
                        for (
                            v
                        ) in (
                            p.vertices
                        ):  # check if all vertices are a member of a vertex group
                            try:
                                w = vertex_group.weight(v)
                                if w >= t:
                                    weight = w
                                else:
                                    weight = None
                            except RuntimeError:  # missing index
                                weight = None
                    else:
                        for (
                            v
                        ) in (
                            p.vertices
                        ):  # check if at least one vertex is a member of a vertex group
                            try:
                                w = vertex_group.weight(v)
                                if w >= t:
                                    weight = w
                                    break
                            except RuntimeError:  # missing index
                                pass
                    if weight:
                        for loop in p.loop_indices:
                            vertex_colors[loop].color[:3] = color.copy()
            else:
                for loop in mesh.loops:
                    try:
                        weight = vertex_group.weight(loop.vertex_index)
                        if weight >= t:
                            vertex_colors[loop.index].color[:3] = color.copy()
                    except RuntimeError:  # missing index
                        pass

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80  scene.update()

        return {"FINISHED"}


def meetsRequirements(context, allselected=False):
    if allselected:
        for ob in context.selected_objects:
            print(ob.name, isinstance(ob, bpy.types.Object), isinstance(ob.data, bpy.types.Mesh))
        return context.mode == "PAINT_VERTEX" and all(
            isinstance(ob, bpy.types.Object)
            and isinstance(ob.data, bpy.types.Mesh)
            and ob.data.color_attributes.active_color_index >= 0
            for ob in context.selected_objects
        )
    return (
        context.mode == "PAINT_VERTEX"
        and isinstance(context.object, bpy.types.Object)
        and isinstance(context.object.data, bpy.types.Mesh)
        and context.object.data.color_attributes.active_color_index >= 0
    )


class NormalizeVertexColors(bpy.types.Operator):
    bl_idname = "mesh.normalize_vertexcolors"
    bl_label = "Normalize Vertex Colors"
    bl_description = "Remap vertex colors to a set of colors that are spaced far apart in the color cube"
    bl_options = {"REGISTER", "UNDO"}

    allselected: BoolProperty(
        name="All selected",
        description="Apply across all selected objects (ignores add-on preferences)",
        default=False,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has an active vertex colors layer.
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and context.object.data.color_attributes.active_color_index >= 0
        )
        return p

    def draw(self, context):
        col = self.layout
        col.prop(self, "allselected")

    def execute(self, context):
        if not meetsRequirements(context, self.allselected):
            self.report(
                {"ERROR"},
                "Not all selected objects are meshes or have a vertex color layer",
            )
            return {"CANCELLED"}
        
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.allselected:
            unique_colors_per_object = {}
            for ob in context.selected_objects:
                mesh = ob.data

                # get a list of unique vertex colors
                vertex_colors = mesh.color_attributes[
                    mesh.color_attributes.active_color_index
                ].data
                nloops = len(vertex_colors)
                loopcolors = np.empty(nloops * 4, dtype=np.float32)
                vertex_colors.foreach_get("color", loopcolors)
                loopcolors.shape = nloops, 4
                unique_colors_per_object[ob] = loopcolors

            # create a list of colors that are unique across objects (so the same RGBA color in two objects are treated as two distinct corners)
            # each entry is a 5-vector [obid,R,G,B,A]
            # we also return an map[ob] -> obid
            unique_colors, obmap = unique_3d(unique_colors_per_object)
            # create a normalized set for all unique colors
            normalized_colors = color_set(len(unique_colors))

            # print("shape of unique_colors", unique_colors.shape)
            # print(unique_colors)
            # print("shape of normalized_colors", normalized_colors.shape)
            # print(normalized_colors)
            # remap the colors
            mcolor = np.empty(5, dtype=np.float32)
            for ob in context.selected_objects:
                mesh = ob.data
                vertex_colors = mesh.color_attributes[
                    mesh.color_attributes.active_color_index
                ].data
                mcolor[0] = obmap[ob]
                for p in mesh.polygons:
                    for i in range(p.loop_start, p.loop_start + p.loop_total):
                        mcolor[1:] = unique_colors_per_object[ob][i]
                        match = np.where(
                            np.all(
                                np.isclose(
                                    unique_colors,
                                    mcolor,
                                    atol=0.0001,
                                ),
                                axis=1,
                            )
                        )[
                            0
                        ]  # yes this returns a list of arrays, the first one is the one we need
                        # print(
                        #    p,
                        #    i,
                        #    mcolor,
                        #    normalized_colors[match[0]] if len(match) else "no match",
                        # )
                        if len(match):
                            unique_colors_per_object[ob][i] = normalized_colors[
                                match[0]
                            ]
                # return the vertex colors to the attribute layer
                vertex_colors.foreach_set(
                    "color", unique_colors_per_object[ob].flatten()
                )

        else:
            ob = context.active_object
            mesh = ob.data

            # get a list of unique vertex colors
            vertex_colors = mesh.color_attributes[
                mesh.color_attributes.active_color_index
            ].data
            nloops = len(vertex_colors)
            loopcolors = np.empty(nloops * 4, dtype=np.float32)
            vertex_colors.foreach_get("color", loopcolors)
            loopcolors.shape = nloops, 4
            unique_colors = unique_2d(loopcolors)

            # create a normalized set
            normalized_colors = color_set(len(unique_colors))

            # remap the colors
            for p in mesh.polygons:
                for i in range(p.loop_start, p.loop_start + p.loop_total):
                    match = np.where(
                        np.all(
                            np.isclose(unique_colors, loopcolors[i], atol=0.0001),
                            axis=1,
                        )
                    )[
                        0
                    ]  # yes this returns a list of arrays, the first one is the one we need
                    if len(match):
                        loopcolors[i] = normalized_colors[match[0]]
            # return the vertex colors
            vertex_colors.foreach_set("color", loopcolors.flatten())

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80 context.scene.update()
        return {"FINISHED"}


def line(put, x0, y0, x1, y1):
    "Bresenham's line algorithm"
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            put(x, y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            put(x, y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    put(x, y)


def tri(put, v0, v1, v2):
    def min3(a, b, c):
        return min(a, min(b, c))

    def max3(a, b, c):
        return max(a, max(b, c))

    def orient2d(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    # Compute triangle bounding box
    minX = min3(v0[0], v1[0], v2[0])
    minY = min3(v0[1], v1[1], v2[1])
    maxX = max3(v0[0], v1[0], v2[0])
    maxY = max3(v0[1], v1[1], v2[1])

    # Rasterize
    for y in range(minY, maxY + 1):
        for x in range(minX, maxX + 1):
            # Determine barycentric coordinates
            # this assumes counter clockwise order
            # w0 = orient2d(v1, v2, x, y)
            # w1 = orient2d(v2, v0, x, y)
            # w2 = orient2d(v0, v1, x, y)
            # tessellate returns tris in clockwise order
            w0 = orient2d(v1, v0, x, y)
            w1 = orient2d(v0, v2, x, y)
            w2 = orient2d(v2, v1, x, y)
            # If p is on or inside all edges, render pixel.
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                put(x, y)


def draw_tris_np(pixels, size, color, tris, step):
    """
    pixels  a size x size x channels  int32 ndarray
    size    size of the image
    color   N x length int32 ndarray (either length 3 or 4)
    tris    a N x 3 x 2 float python list of lists of lists
    step    a function thats is called with argument (1) every 100 triangles
    """

    XY = np.array([size - 1, size - 1], dtype=np.int32)

    N = len(tris)
    # create a scaled list of triangles w. coords cast to ints
    v = np.empty((N, 3, 2), dtype=np.int32)
    v[:] = tris * XY

    # Compute triangle bounding box
    minXY = np.min(v, axis=1)
    maxXY = np.max(v, axis=1)

    # Rasterize
    # Create xy vectors for all points in the gris
    xy = np.empty((size, size, 2), dtype=np.int32)
    xy[:, :, 1], xy[:, :, 0] = np.meshgrid(
        np.arange(size, dtype=np.int32), np.arange(size, dtype=np.int32)
    )

    # Calculate the edge vectors
    tri0m1 = v[:, 0] - v[:, 1]
    tri2m0 = v[:, 2] - v[:, 0]
    tri1m2 = v[:, 1] - v[:, 2]

    # Process each triangle
    for i in range(N):
        if i % 100 == 1:
            step(1)
        tri = v[i]
        # determine bounding box
        minX = minXY[i, 0]
        minY = minXY[i, 1]
        maxX = maxXY[i, 0]
        maxY = maxXY[i, 1]
        xyv = xy[minX : maxX + 1, minY : maxY + 1]
        # calculate barycentric coords
        w0 = tri0m1[i, 0] * (xyv[:, :, 1] - tri[1, 1]) - (tri0m1[i, 1]) * (
            xyv[:, :, 0] - tri[1, 0]
        )
        w1 = tri2m0[i, 0] * (xyv[:, :, 1] - tri[0, 1]) - (tri2m0[i, 1]) * (
            xyv[:, :, 0] - tri[0, 0]
        )
        w2 = tri1m2[i, 0] * (xyv[:, :, 1] - tri[2, 1]) - (tri1m2[i, 1]) * (
            xyv[:, :, 0] - tri[2, 0]
        )
        # check if point is inside triangle
        w = np.logical_and(np.logical_and(w0 >= 0, w1 >= 0), w2 >= 0).T
        # assign color for those points that are
        pixelsview = pixels[minY : maxY + 1, minX : maxX + 1]
        pixelsview[w] = color[i]


class VertexColorToImageMap(bpy.types.Operator):
    bl_idname = "mesh.vertexccolortoimagemap"
    bl_label = "Bake Vertex Color"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Create an image map for the active vertex color layer using the active uv-map, possibly restricted to selected faces or faces with a materialid equal to the active material"

    size: EnumProperty(
        name="Size",
        description="square size of imagemap",
        items=[
            ("1024", "1024x1024", "a 1K map"),
            ("2048", "2048x2048", "a 2K map"),
            ("4096", "4096x4096", "a 4K map"),
        ],
    )
    bleed: BoolProperty(
        name="Add margin",
        description="add extra pixel around baked colors",
        default=True,
    )
    materialid: BoolProperty(
        name="Active Material only",
        description="Only bake facemap colors for faces with the active material id",
        default=False,
    )
    selected: BoolProperty(
        name="Selected faces only",
        description="Only bake facemap colors for faces that are selected",
        default=False,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh.
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(  # actually other modes would be fine
                context.object, bpy.types.Object
            )
            and isinstance(context.object.data, bpy.types.Mesh)
        )
        return p

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "bleed")
        layout.prop(self, "materialid")
        layout.prop(self, "selected")
        layout.prop(self, "size")

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        self.ob = context.active_object
        mesh = context.object.data

        # select the active uv layer or create one if it dows not exist yet
        if mesh.uv_layers.active is None:
            bpy.ops.mesh.uv_texture_add()
            # unwrap the mesh
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(
                action="SELECT"
            )  # make sure all verts are selected to get a complete map
            bpy.ops.uv.smart_project(island_margin=0.03)
            bpy.ops.object.mode_set(mode="OBJECT")
            mesh = context.object.data
        uv = mesh.uv_layers.active.data

        # select the active vertex color layer or create one if it does not exist yet
        if mesh.color_attributes.active_color_index < 0:
            bpy.ops.geometry.color_attribute_add(
                domain="CORNER", data_type="BYTE_COLOR"
            )
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        size = int(self.size)
        img = bpy.data.images.new(name="IDmap " + mesh.name, width=size, height=size)
        img.use_fake_user = True

        start = time()
        # draw polygons onto image
        pixels = np.asarray(img.pixels, dtype=np.float32)
        channels = img.channels
        tris = []
        colors = []
        # TODO? materialid AND selected
        if self.materialid:
            matid = self.ob.active_material_index
            for p in mesh.polygons:
                if p.material_index == matid:
                    polygon = []
                    color = vertex_colors[p.loop_start].color
                    for loop in p.loop_indices:
                        polygon.append(uv[loop].uv)  # uv is 2-vector
                    for tri in tessellate([polygon]):
                        tris.append((polygon[tri[0]], polygon[tri[1]], polygon[tri[2]]))
                        colors.append(color[:3])
        elif self.selected:
            for p in mesh.polygons:
                if p.select:
                    polygon = []
                    color = vertex_colors[p.loop_start].color
                    for loop in p.loop_indices:
                        polygon.append(uv[loop].uv)  # uv is 2-vector
                    for tri in tessellate([polygon]):
                        tris.append((polygon[tri[0]], polygon[tri[1]], polygon[tri[2]]))
                        colors.append(color[:3])
        else:
            for p in mesh.polygons:
                polygon = []
                color = vertex_colors[p.loop_start].color
                for loop in p.loop_indices:
                    polygon.append(uv[loop].uv)  # uv is 2-vector
                for tri in tessellate([polygon]):
                    tris.append((polygon[tri[0]], polygon[tri[1]], polygon[tri[2]]))
                    colors.append(color[:3])

        pixels.shape = size, size, channels
        vcolors = np.ones((len(colors), 4), dtype=np.float32)
        vcolors[:, :3] = colors
        with ProgressCM(wm=context.window_manager, steps=len(tris) // 100) as progress:
            draw_tris_np(pixels, size, vcolors, tris, progress.step)
        if self.bleed:
            black = np.array([0, 0, 0, 1], dtype=np.float32)
            ct = pixels[1:-1, 1:-1]
            up = pixels[1:-1, 0:-2]
            dn = pixels[1:-1, 2:]
            lf = pixels[0:-2, 1:-1]
            ri = pixels[2:, 1:-1]
            ctb = np.all(ct == black, axis=-1)
            edge = np.logical_and(ctb, np.logical_not(np.all(up == black, axis=-1)))
            edge = np.dstack((edge, edge, edge, edge))
            np.putmask(ct, edge, up)
            edge = np.logical_and(ctb, np.logical_not(np.all(dn == black, axis=-1)))
            edge = np.dstack((edge, edge, edge, edge))
            np.putmask(ct, edge, dn)
            edge = np.logical_and(ctb, np.logical_not(np.all(lf == black, axis=-1)))
            edge = np.dstack((edge, edge, edge, edge))
            np.putmask(ct, edge, lf)
            edge = np.logical_and(ctb, np.logical_not(np.all(ri == black, axis=-1)))
            edge = np.dstack((edge, edge, edge, edge))
            np.putmask(ct, edge, ri)
        pixels.shape = -1
        img.pixels[:] = pixels
        self.report(
            {"INFO"},
            "%d tris on a %s grid took %.1fs" % (len(tris), self.size, time() - start),
        )
        img.update()

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        # we cannot stay in edit mode to show uv-map on top of image because we would lose the size option
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80 scene.update()
        # make sure we have an image editor visible
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                break
        else:
            area = context.area
        area.type = "IMAGE_EDITOR"
        # and let it show the new image
        for space in area.spaces:
            if space.type == "IMAGE_EDITOR":
                space.image = img
        area.tag_redraw()
        return {"FINISHED"}


class ImageMapCombine(bpy.types.Operator):
    bl_idname = "mesh.imagemapcombine"
    bl_label = "Combine Image Maps"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Combine two image maps into a third"

    image1: StringProperty(name="Image 1", default="")
    image2: StringProperty(name="Image 2", default="")

    @classmethod
    def poll(self, context):
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "image1", bpy.data, "images")
        layout.prop_search(self, "image2", bpy.data, "images")

    def execute(self, context):
        if self.image1 in bpy.data.images and self.image2 in bpy.data.images:
            img1 = bpy.data.images[self.image1]
            size1 = tuple(img1.size)
            img2 = bpy.data.images[self.image2]
            size2 = tuple(img2.size)
            if size1 == size2:
                img = bpy.data.images.new(
                    name="Result", width=size1[0], height=size1[1]
                )
                # img.use_fake_user=True
                pixels1 = np.asarray(img1.pixels, dtype=np.float32)
                pixels2 = np.asarray(img2.pixels, dtype=np.float32)
                pixels1 = np.maximum(pixels1, pixels2)
                img.pixels[:] = pixels1
                img.update()
                for area in context.screen.areas:
                    if area.type == "IMAGE_EDITOR":
                        break
                else:
                    area = context.area
                area.type = "IMAGE_EDITOR"
                # and let it show the new image
                for space in area.spaces:
                    if space.type == "IMAGE_EDITOR":
                        space.image = img
                area.tag_redraw()
            else:
                self.report({"WARNING"}, "select two images of equal size")
        else:
            self.report({"WARNING"}, "select two images")
        return {"FINISHED"}


def vector_is_close(a, b, tol):
    return all([abs(c - d) <= tol for c, d in zip(a, b)])


class VertexColorsToMaterials(bpy.types.Operator):
    bl_idname = "mesh.vertexcolorstomaterials"
    bl_label = "Vertex Colors To Materials"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = "Create a material assignment for each unique vertex color"

    tolerance: FloatProperty(
        name="Tolerance",
        description="Per color channel tolerance in matching colors",
        min=0.0,
        max=1.0,
        step=1,
        default=0.01,
    )
    match: BoolProperty(
        name="Match Color List",
        description="Try to use names from color list for material names",
        default=True,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and we have a suitable render engine
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and context.scene.render.engine
            in {"BLENDER_EEVEE", "CYCLES", "BLENDER_RENDER"}
        )
        return p

    def draw(self, context):
        col = self.layout
        col.prop(self, "match")
        if self.match:
            col.prop(self, "tolerance")

    def execute(self, context):
        bpy.ops.object.mode_set(mode="OBJECT")

        scene = context.scene
        ob = context.active_object
        mesh = context.object.data

        # select the active vertex color layer or create one if it does not exist yet
        if mesh.color_attributes.active_color_index < 0:
            bpy.ops.geometry.color_attribute_add(
                domain="CORNER", data_type="BYTE_COLOR"
            )
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data

        materialmap = {}
        for p in mesh.polygons:
            for loop in p.loop_indices:
                color = vertex_colors[loop].color[:3]
                break  # we only check the first vertex, i.e. assume the face is uniform
            # color.freeze()
            if color not in materialmap:
                bpy.ops.object.material_slot_add()
                matid = ob.active_material_index
                slot = ob.material_slots[matid]
                slot.link = "DATA"
                matname = "IDMap"
                if self.match:
                    for item in context.scene.colormap:
                        if vector_is_close(color, item.color, self.tolerance):
                            matname = item.name
                mat = bpy.data.materials.new(matname)
                slot.material = mat
                mat.diffuse_color[:3] = color[:3]
                mat.use_nodes = True  # change it after creating a new slot so it will inherit the viewport color as the diffuse
                # NOTE might be bug here: once a single material hase use_nodes that seems to be the default for new ones...
                materialmap[color] = matid
                p.material_index = matid
                if context.scene.render.engine == "BLENDER_RENDER":
                    mat.node_tree.nodes[
                        "Material"
                    ].material = mat  # strange circular dependance in Blender internal
                nodes = mat.node_tree.nodes
                principled = next(
                    (
                        n
                        for n in nodes
                        if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)
                    ),
                    None,
                )
                principled.inputs[0].default_value = [color[0], color[1], color[2], 1.0]
            else:
                p.material_index = materialmap[color]

        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="VERTEX_PAINT")
        # no longer present in 2.80 scene.update()

        return {"FINISHED"}


### Color List management
class IDMapPanel(bpy.types.Panel):
    bl_idname = "PAINT_PT_IDMap"
    bl_label = "ID Color List"
    bl_options = {"DEFAULT_CLOSED"}
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ID Color List"

    @classmethod
    def poll(cls, context):
        return context.mode == "PAINT_VERTEX"

    def draw_header(self, context):
        layout = self.layout

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()
        col.template_list(
            "COLORMAP_UL_IDMapper",
            "",
            context.scene,
            "colormap",
            context.scene,
            "active_colormap",
            rows=5,
        )
        brow = col.row()
        brow.operator("scene.import_id_map", text="Load", icon="FILEBROWSER")
        save = "Save"
        if context.scene.colormap_changed:
            save += " *"
        brow.operator("scene.export_id_map", text=save, icon="FILE_TICK")
        brow.operator("scene.colormap_initialize", text="Init", icon="NEWFOLDER")
        col = row.column(align=True)
        col.operator("scene.colormap_add", icon="ADD", text="")
        col.operator("scene.colormap_remove", icon="REMOVE", text="")
        # col.menu("MESH_MT_vertex_group_specials", icon='DOWNARROW_HLT', text="")
        col.separator()
        col.operator("scene.colormap_move_up", icon="TRIA_UP", text="")
        col.operator("scene.colormap_roll_up", icon="LOOP_BACK", text="")
        col.operator("scene.colormap_move_down", icon="TRIA_DOWN", text="")


### Operator utility panel
class IDMapOpsPanel(bpy.types.Panel):
    bl_idname = "PAINT_PT_IDMapperops"
    bl_label = "ID Mapper utils"
    bl_options = set()
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ID Tools"

    @classmethod
    def poll(cls, context):
        return context.mode == "PAINT_VERTEX"

    def draw_header(self, context):
        layout = self.layout

    def draw(self, context):
        col = self.layout.column()
        if "icons" in preview_collections:
            col.operator(
                IDMapper.bl_idname,
                icon_value=preview_collections["icons"]["idmapper_icon"].icon_id,
            )
            col.operator(
                VertexColorMerge.bl_idname,
                icon_value=preview_collections["icons"]["vcolmerge_icon"].icon_id,
            )
            col.operator(
                VertexColorFromSelected.bl_idname,
                icon_value=preview_collections["icons"]["selected2vcol_icon"].icon_id,
            )
            col.operator(
                VertexColorCopy.bl_idname,
                icon_value=preview_collections["icons"]["vcolcopy_icon"].icon_id,
            )
            col.operator(
                VertexColorFacePaint.bl_idname,
                icon_value=preview_collections["icons"]["facepaint_icon"].icon_id,
            )
            col.operator(
                VGroupsToVertexColor.bl_idname,
                icon_value=preview_collections["icons"]["vgroup2vcol_icon"].icon_id,
            )
            col.operator(
                VertexColorsToMaterials.bl_idname,
                icon_value=preview_collections["icons"]["vcol2mat_icon"].icon_id,
            )
            col.operator(
                VertexColorToImageMap.bl_idname,
                icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
            )
            col.operator(
                VertexColorToImageMap.bl_idname,
                text="Bake Vertex Color (selected)",
                icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
            ).selected = True
            col.operator(
                VertexColorToImageMap.bl_idname,
                text="Bake Vertex Color (active material id)",
                icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
            ).materialid = True
            col.operator(
                NormalizeVertexColors.bl_idname,
                icon_value=preview_collections["icons"]["vcolnormalize_icon"].icon_id,
            )
        else:
            col.operator(IDMapper.bl_idname, icon="PLUGIN")
            col.operator(VertexColorMerge.bl_idname, icon="PLUGIN")
            col.operator(VertexColorFromSelected.bl_idname, icon="PLUGIN")
            col.operator(VertexColorCopy.bl_idname, icon="PLUGIN")
            col.operator(VertexColorFacePaint.bl_idname, icon="PLUGIN")
            col.operator(VGroupsToVertexColor.bl_idname, icon="PLUGIN")
            col.operator(VertexColorsToMaterials.bl_idname, icon="PLUGIN")
            col.operator(VertexColorToImageMap.bl_idname, icon="PLUGIN")
            col.operator(
                VertexColorToImageMap.bl_idname,
                text="Bake Vertex Color (selected)",
                icon="PLUGIN",
            ).selected = True
            col.operator(
                VertexColorToImageMap.bl_idname,
                text="Bake Vertex Color (active material id)",
                icon="PLUGIN",
            ).materialid = True
            col.operator(NormalizeVertexColors.bl_idname, icon="PLUGIN")


def names_unique(colormap):
    names = {item.name for item in colormap}
    return len(names) == len(colormap)


PLACEHOLDER = "@@@unlikely_to_be_ever_chosen@@@"


def unique_name(colormap, name):
    if name == PLACEHOLDER:
        return name
    names = Counter(item.name for item in colormap)
    if names[name] > 1:
        while name in names:
            m = match(r"^(.*?\.)(\d+)$", name)
            if m:
                name = m.group(1) + format(int(m.group(2)) + 1, "03d")
            else:
                name += ".001"
    return name


def idcl_changed(self, context):
    context.scene.colormap_changed = True


depth = 0


def make_unique(self, context):
    global depth
    idcl_changed(self, context)
    # the set= accessor on properties does not allow us to set the underlying value
    # the update= trigger does but we have to prevent endless recursion in that case because
    # assigning will cause a second trigger
    # we also bail out if duplicates are ok
    if depth or context.preferences.addons[__name__].preferences.allow_duplicate_names:
        return
    depth += 1
    self.name = unique_name(context.scene.colormap, self.name)
    depth -= 1


def index_changed(self, context):
    if context.scene.active_colormap < len(context.scene.colormap):
        context.tool_settings.vertex_paint.brush.color = tuple(
            context.scene.colormap[context.scene.active_colormap].color[:3]
        )


class ColorMap(bpy.types.PropertyGroup):
    color: bpy.props.FloatVectorProperty(
        name="Color",
        default=(0.0, 0.0, 0.0),
        subtype="COLOR_GAMMA",
        size=3,
        min=0,
        max=1,
        update=idcl_changed,
        get=None,
        set=None,
    )
    name: bpy.props.StringProperty(name="Name", default="Material", update=make_unique)


class COLORMAP_UL_IDMapper(
    bpy.types.UIList
):  # change to UI_UL_list later for sorting and filtering
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        colormap = item
        if self.layout_type in {
            "DEFAULT",
            "COMPACT",
            "GRID",
        }:  # we basically ignore the distinction
            row = layout.row()
            row.alignment = "LEFT"
            rowi = row.row()
            rowi.label(text="", icon="COLOR")
            rowi.prop(colormap, "name", text="", emboss=False)
            rowi.prop(colormap, "color", text="", emboss=True)
            row.label(text=str(index + 1))


class SceneColormapAdd(bpy.types.Operator):
    bl_idname = "scene.colormap_add"
    bl_label = "Add Color List Entry"
    bl_description = "Add Color List Entry"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    @classmethod
    def poll(self, context):
        return True

    def execute(self, context):
        cm = context.scene.colormap.add()
        cm.color = tuple(context.tool_settings.vertex_paint.brush.color)
        context.scene.active_colormap = len(context.scene.colormap) - 1
        context.scene.colormap_changed = True
        return {"FINISHED"}


class SceneColormapSetActive(bpy.types.Operator):
    bl_idname = "scene.colormap_set_active"
    bl_label = "Set Active Color List Entry"
    bl_description = "Set Active Color List Entry"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    index: bpy.props.IntProperty(name="Index", default=0, min=0)

    @classmethod
    def poll(self, context):
        return True

    def execute(self, context):
        if self.index < len(context.scene.colormap):
            context.scene.active_colormap = self.index
        return {"FINISHED"}


class SceneColormapMoveUp(bpy.types.Operator):
    bl_idname = "scene.colormap_move_up"
    bl_label = "Move Color List Entry Up"
    bl_description = "Move Color List Entry Up"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    @classmethod
    def poll(self, context):
        return (
            context.scene.active_colormap > 0
        )  # implies we also have at least 2 items

    def execute(self, context):
        index = context.scene.active_colormap
        colormap = context.scene.colormap
        a = colormap[index - 1]
        b = colormap[index]
        aname, bname = a.name, b.name
        a.name, b.name = PLACEHOLDER, PLACEHOLDER
        a.name, b.name = bname, aname
        a.color, b.color = tuple(b.color), tuple(
            a.color
        )  # FloatArray is wrapped so we need copy
        context.scene.active_colormap -= 1
        return {"FINISHED"}


class SceneColormapRollUp(bpy.types.Operator):
    bl_idname = "scene.colormap_roll_up"
    bl_label = "Roll Color List Up"
    bl_description = "Roll Color List Up"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    @classmethod
    def poll(self, context):
        return len(context.scene.colormap) > 1

    def execute(self, context):
        originalindex = context.scene.active_colormap
        colormap = context.scene.colormap
        for index in range(len(colormap) - 1):
            a = colormap[index - 1]
            b = colormap[index]
            aname, bname = a.name, b.name
            a.name, b.name = PLACEHOLDER, PLACEHOLDER
            a.name, b.name = bname, aname
            a.color, b.color = tuple(b.color), tuple(
                a.color
            )  # FloatArray is wrapped so we need copy
        context.scene.active_colormap = originalindex
        return {"FINISHED"}


class SceneColormapMoveDown(bpy.types.Operator):
    bl_idname = "scene.colormap_move_down"
    bl_label = "Move Color List Entry Down"
    bl_description = "Move Color List Entry Down"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    @classmethod
    def poll(self, context):
        return context.scene.active_colormap < len(context.scene.colormap) - 1

    def execute(self, context):
        index = context.scene.active_colormap
        colormap = context.scene.colormap
        a = colormap[index + 1]
        b = colormap[index]
        aname, bname = a.name, b.name
        a.name, b.name = PLACEHOLDER, PLACEHOLDER
        a.name, b.name = bname, aname
        a.color, b.color = tuple(b.color), tuple(
            a.color
        )  # FloatArray is wrapped so we need copy
        context.scene.active_colormap += 1
        return {"FINISHED"}


class SceneColormapRemove(bpy.types.Operator):
    bl_idname = "scene.colormap_remove"
    bl_label = "Remove Color List Entry"
    bl_description = "Remove Color List Entry"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    @classmethod
    def poll(self, context):
        return len(context.scene.colormap) > 1  # we always keep at least one entry

    def execute(self, context):
        originalindex = context.scene.active_colormap
        context.scene.colormap.remove(context.scene.active_colormap)
        context.scene.active_colormap = originalindex - 1 if originalindex else 0
        context.scene.colormap_changed = True
        return {"FINISHED"}


class SceneColormapInitialize(bpy.types.Operator):
    bl_idname = "scene.colormap_initialize"
    bl_label = "Initialize Color List"
    bl_description = "Add Color List entries for each unique vertex color"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    tolerance: FloatProperty(
        name="Tolerance",
        description="Per color channel tolerance in matching colors",
        min=0.0,
        max=1.0,
        step=1,
        default=0.01,
    )
    noduplicates: BoolProperty(
        name="No duplicates",
        description="do not create new items if existing item matches color",
        default=True,
    )

    @classmethod
    def poll(self, context):
        """
        Only visible in vertex paint mode if the active object is a mesh and has vertex groups.
        """
        p = (
            context.mode == "PAINT_VERTEX"
            and isinstance(context.object, bpy.types.Object)
            and isinstance(context.object.data, bpy.types.Mesh)
            and len(context.object.vertex_groups) > 0
        )
        return True

    def draw(self, context):
        col = self.layout
        col.prop(self, "noduplicates")
        if self.noduplicates:
            col.prop(self, "tolerance")

    def execute(self, context):
        ob = context.active_object
        mesh = ob.data
        vertex_colors = mesh.color_attributes[
            mesh.color_attributes.active_color_index
        ].data
        nloops = len(vertex_colors)
        loopcolors = np.empty(nloops * 4, dtype=np.float32)
        vertex_colors.foreach_get("color", loopcolors)
        loopcolors.shape = nloops, 4
        unique_colors = unique_2d(loopcolors)
        for color in unique_colors:
            doadd = True
            if self.noduplicates:
                for item in context.scene.colormap:
                    if vector_is_close(color, item.color, self.tolerance):
                        doadd = False
            if doadd:
                cm = context.scene.colormap.add()
                cm.color = Vector(color[:3])
        context.scene.active_colormap = len(context.scene.colormap) - 1

        context.scene.colormap_changed = True
        return {"FINISHED"}


class ExportIDMap(bpy.types.Operator, ExportHelper):
    bl_idname = "scene.export_id_map"
    bl_label = "Export ID Map"
    bl_description = "Export ID Map to file"
    filename_ext = ".csv"
    filter_glob: bpy.props.StringProperty(default="*.csv", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        filepath = self.filepath
        filepath = bpy.path.ensure_ext(filepath, self.filename_ext)
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            for item in context.scene.colormap:
                writer.writerow(
                    [item.name] + [format(c, ".4f") for c in item.color]
                )  # should apply %.4f
            context.scene.colormap_changed = False
        return {"FINISHED"}


def hexcolor(s):
    color = [1, 1, 1]
    m = match(r"\s*#?([\da-fA-F]{2})([\da-fA-F]{2})([\da-fA-F]{2})\s*", s)
    if m:
        for i in (1, 2, 3):
            color[i - 1] = int(m.group(i), base=16) / 255
    return color


class ImportIDMap(bpy.types.Operator, ImportHelper):
    bl_idname = "scene.import_id_map"
    bl_label = "Import ID Map"
    bl_description = "Import ID Map from file"
    bl_options = {"UNDO"}

    files: bpy.props.CollectionProperty(
        name="File Path",
        description="File path used for importing " "the Color List file",
        type=bpy.types.OperatorFileListElement,
    )

    directory: bpy.props.StringProperty()

    filename_ext = ".csv"
    filter_glob: bpy.props.StringProperty(default="*.csv", options={"HIDDEN"})

    def execute(self, context):
        paths = [join(self.directory, name.name) for name in self.files]
        if not paths:
            paths.append(self.filepath)

        duplicates = False
        context.scene.colormap.clear()
        context.scene.active_colormap = 0
        for path in paths:
            with open(path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    print(row)
                    item = context.scene.colormap.add()
                    item.name = row[0]
                    duplicates |= item.name != row[0]
                    if len(row) > 2:
                        color = [1, 1, 1]
                        for n, c in enumerate(row[1:]):
                            if n > 2:
                                break
                            color[n] = float(c)
                        item.color = color
                    else:
                        item.color = hexcolor(row[1])
            break  # ignore any extra paths
        settings = context.preferences.addons[__name__].preferences
        if duplicates:
            self.report({"WARNING"}, "Names in color list were not unique")
        index_changed(self, context)  # make sure current brush color gets set
        context.scene.colormap_changed = False
        return {"FINISHED"}


### End Color List management


def load_icons():
    import os
    import bpy
    import bpy.utils

    try:  # if anything goes wrong, for example because we are not running 2.75+ we just ignore it
        import bpy.utils.previews

        pcoll = bpy.utils.previews.new()

        # path to the folder where the icon is
        # the path is calculated relative to this py file inside the addon folder
        my_icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        pcoll.load("idmapper_icon", os.path.join(my_icons_dir, "IDMapper.png"), "IMAGE")
        pcoll.load(
            "facepaint_icon", os.path.join(my_icons_dir, "Facepaint.png"), "IMAGE"
        )
        pcoll.load(
            "selected2vcol_icon",
            os.path.join(my_icons_dir, "Selected2Vcol.png"),
            "IMAGE",
        )
        pcoll.load("vcolcopy_icon", os.path.join(my_icons_dir, "Vcolcopy.png"), "IMAGE")
        pcoll.load(
            "vcolmerge_icon", os.path.join(my_icons_dir, "Vcolmerge.png"), "IMAGE"
        )
        pcoll.load(
            "vgroup2vcol_icon", os.path.join(my_icons_dir, "Vgroup2Vcol.png"), "IMAGE"
        )
        pcoll.load("vcolbake_icon", os.path.join(my_icons_dir, "Vcolbake.png"), "IMAGE")
        pcoll.load(
            "vcol2mat_icon", os.path.join(my_icons_dir, "Vcoltomaterials.png"), "IMAGE"
        )
        pcoll.load(
            "vcolnormalize_icon",
            os.path.join(my_icons_dir, "Vcolnormalize.png"),
            "IMAGE",
        )

        preview_collections["icons"] = pcoll
    except Exception as e:
        # print(e)
        pass


addon_keymaps = []


def menu_func_vcol(self, context):
    self.layout.separator()
    if "icons" in preview_collections:
        self.layout.operator(
            IDMapper.bl_idname,
            icon_value=preview_collections["icons"]["idmapper_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorMerge.bl_idname,
            icon_value=preview_collections["icons"]["vcolmerge_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorFromSelected.bl_idname,
            icon_value=preview_collections["icons"]["selected2vcol_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorCopy.bl_idname,
            icon_value=preview_collections["icons"]["vcolcopy_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorFacePaint.bl_idname,
            icon_value=preview_collections["icons"]["facepaint_icon"].icon_id,
        )
        self.layout.operator(
            VGroupsToVertexColor.bl_idname,
            icon_value=preview_collections["icons"]["vgroup2vcol_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorsToMaterials.bl_idname,
            icon_value=preview_collections["icons"]["vcol2mat_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorToImageMap.bl_idname,
            icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
        )
        self.layout.operator(
            VertexColorToImageMap.bl_idname,
            text="Bake Vertex Color (selected)",
            icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
        ).selected = True
        self.layout.operator(
            VertexColorToImageMap.bl_idname,
            text="Bake Vertex Color (active material id)",
            icon_value=preview_collections["icons"]["vcolbake_icon"].icon_id,
        ).materialid = True
    else:
        self.layout.operator(IDMapper.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorMerge.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorFromSelected.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorCopy.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorFacePaint.bl_idname, icon="PLUGIN")
        self.layout.operator(VGroupsToVertexColor.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorsToMaterials.bl_idname, icon="PLUGIN")
        self.layout.operator(VertexColorToImageMap.bl_idname, icon="PLUGIN")
        self.layout.operator(
            VertexColorToImageMap.bl_idname,
            text="Bake Vertex Color (selected)",
            icon="PLUGIN",
        ).selected = True
        self.layout.operator(
            VertexColorToImageMap.bl_idname,
            text="Bake Vertex Color (active material id)",
            icon="PLUGIN",
        ).materialid = True


def menu_func_image(self, context):
    self.layout.separator()
    self.layout.operator(ImageMapCombine.bl_idname, icon="PLUGIN")


classes = (
    IDMapperPrefs,
    IDMapper,
    VertexColorMerge,
    VertexColorFromSelected,
    VertexColorCopy,
    VertexColorFacePaint,
    VGroupsToVertexColor,
    NormalizeVertexColors,
    VertexColorToImageMap,
    IDMapPanel,
    IDMapOpsPanel,
    VertexColorsToMaterials,
    ColorMap,
    COLORMAP_UL_IDMapper,
    SceneColormapAdd,
    SceneColormapSetActive,
    SceneColormapMoveUp,
    SceneColormapRollUp,
    SceneColormapMoveDown,
    SceneColormapRemove,
    SceneColormapInitialize,
    ExportIDMap,
    ImportIDMap,
    ImageMapCombine,
)

register_classes, unregister_classes = bpy.utils.register_classes_factory(classes)


def register():
    load_icons()
    register_classes()
    bpy.types.Scene.colormap = bpy.props.CollectionProperty(type=ColorMap)
    bpy.types.Scene.active_colormap = bpy.props.IntProperty(
        name="Active", default=0, update=index_changed
    )
    bpy.types.Scene.colormap_changed = bpy.props.BoolProperty(
        name="Color List Changed", default=False
    )
    bpy.types.VIEW3D_MT_paint_vertex.append(menu_func_vcol)
    bpy.types.IMAGE_MT_image.append(menu_func_image)

    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name="Vertex Paint", space_type="EMPTY")
    kmi = km.keymap_items.new(
        VertexColorFacePaint.bl_idname,
        "P",
        "RELEASE",
        ctrl=False,
        shift=False,
        head=True,
    )
    addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    try:
        for pcoll in preview_collections.values():
            bpy.utils.previews.remove(pcoll)
    except Exception as e:
        print(e)
        pass
    bpy.types.IMAGE_MT_image.remove(menu_func_image)
    bpy.types.VIEW3D_MT_paint_vertex.remove(menu_func_vcol)
    unregister_classes()


if __name__ == "__main__":
    register()