# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
#     "numpy",
# ]
# ///
"""Ray-trace all-atom ESMFold conformations colored by NeuroEGNN co-attention in PyMOL."""

import os
import json
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_DIR, "results", "figure_data", "case_3d_data.npz")
PDB_DIR = os.path.join(CODE_DIR, "results", "figure_data", "esmfold_pdbs")
OUT_DIR = os.path.join(CODE_DIR, "results", "pymol_sessions")
os.makedirs(OUT_DIR, exist_ok=True)

d = np.load(DATA_PATH, allow_pickle=True)
VIEWS_META = {}


def compute_comb_norm(s2g, g2s):
    L = s2g.shape[0]
    comb = 0.5 * ((s2g * L).mean(axis=0) + (g2s * L).mean(axis=0))
    return comb / np.mean(comb)


def setup_render_style():
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("depth_cue", 0)
    cmd.set("specular", 0.26)
    cmd.set("ambient", 0.44)
    cmd.set("direct", 0.60)
    cmd.set("two_sided_lighting", 1)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 0.05)
    cmd.set("ray_shadows", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 0)
    cmd.set("cartoon_side_chain_helper", 1)
    cmd.set("cartoon_loop_radius", 0.24)
    cmd.set("cartoon_oval_length", 1.18)
    cmd.set("cartoon_oval_width", 0.28)
    cmd.set("stick_radius", 0.18)

    cmd.set_color("attr_c1", [203 / 255, 213 / 255, 225 / 255])
    cmd.set_color("attr_c2", [142 / 255, 165 / 255, 188 / 255])
    cmd.set_color("attr_c3", [215 / 255, 185 / 255, 142 / 255])
    cmd.set_color("attr_c4", [192 / 255, 122 / 255, 104 / 255])
    cmd.set_color("attr_c5", [145 / 255, 65 / 255, 83 / 255])
    cmd.set_color("sulf_gold", [214 / 255, 162 / 255, 48 / 255])
    cmd.set_color("het_n", [58 / 255, 108 / 255, 156 / 255])
    cmd.set_color("het_o", [186 / 255, 74 / 255, 74 / 255])
    cmd.set_color("surf_tint", [232 / 255, 239 / 255, 246 / 255])
    cmd.set_color("ghost_gray", [214 / 255, 221 / 255, 230 / 255])


def apply_attr_bfactor_and_color(obj_name, attr_vals):
    for i, val in enumerate(attr_vals, start=1):
        v_clip = float(np.clip(val, 0.45, 2.75))
        cmd.alter(f"{obj_name} and resi {i}", f"b={v_clip:.4f}")
    cmd.rebuild(obj_name)
    cmd.spectrum("b", "attr_c1 attr_c2 attr_c3 attr_c4 attr_c5", obj_name, minimum=0.45, maximum=2.75)


def style_key_sticks(obj_name, resi_list):
    sel = f"{obj_name} and resi {'+'.join(str(r) for r in resi_list)} and (sidechain or name CA)"
    cmd.show("sticks", sel)
    cmd.color("het_n", f"({sel}) and elem N")
    cmd.color("het_o", f"({sel}) and elem O")
    cmd.color("sulf_gold", f"({sel}) and elem S")
    cmd.hide("sticks", f"{obj_name} and elem H")


def record_residue_2d_coords(tag, obj_name, width, height):
    """Project C-alpha and side-chain tip coordinates into normalized 2D image coordinates."""
    v = cmd.get_view()
    R = np.array(v[0:9]).reshape(3, 3)
    cam_pos = np.array(v[9:12])
    origin = np.array(v[12:15])
    fov = cmd.get_setting_float("field_of_view")
    half_h = abs(cam_pos[2]) * np.tan(np.deg2rad(fov * 0.5))
    half_w = half_h * (width / float(height))

    model = cmd.get_model(obj_name)
    res_pts = {}
    for at in model.atom:
        resi = int(at.resi)
        p_mod = np.array(at.coord) - origin
        p_cam = p_mod @ R
        u = 0.5 + (p_cam[0] / (2.0 * half_w))
        v_y = 0.5 + (p_cam[1] / (2.0 * half_h))
        if at.name == "CA":
            res_pts[f"CA_{resi}"] = [float(u), float(v_y)]
        if at.name in ("SG", "CZ", "NZ", "CE2", "SD", "CG2"):
            res_pts[f"TIP_{resi}"] = [float(u), float(v_y)]
    VIEWS_META[tag] = {"width": width, "height": height, "pts": res_pts}


def render_panel_A():
    cmd.reinitialize()
    setup_render_style()
    cmd.load(os.path.join(PDB_DIR, "test_608.pdb"), "pep608")
    assert cmd.count_atoms("pep608") > 0

    attr = compute_comb_norm(d["s2g_608"], d["g2s_608"])
    cmd.hide("everything", "pep608")
    cmd.show("cartoon", "pep608")
    apply_attr_bfactor_and_color("pep608", attr)

    cmd.bond("pep608 and resi 6 and name SG", "pep608 and resi 11 and name SG")
    style_key_sticks("pep608", [6, 7, 11, 13, 20])
    cmd.show("spheres", "pep608 and resi 6+7+11+20 and name SG")
    cmd.set("sphere_scale", 0.25, "pep608 and name SG")

    cmd.show("surface", "pep608")
    cmd.set("surface_color", "surf_tint", "pep608")
    cmd.set("transparency", 0.78, "pep608")

    cmd.orient("pep608")
    cmd.turn("z", 8)
    cmd.turn("x", -10)
    cmd.zoom("pep608", buffer=3.2)

    W, H = 1600, 1120
    record_residue_2d_coords("A", "pep608", W, H)
    cmd.save(os.path.join(OUT_DIR, "Test_608_NeuroEGNN_attr.pdb"), "pep608")
    cmd.save(os.path.join(OUT_DIR, "panelA_test608.pse"))
    cmd.ray(W, H)
    cmd.png(os.path.join(OUT_DIR, "panelA_test608.png"), dpi=300)
    print("Rendered panelA_test608.png")


def render_panel_B():
    cmd.reinitialize()
    setup_render_style()
    cmd.load(os.path.join(PDB_DIR, "test_216.pdb"), "pep216")
    assert cmd.count_atoms("pep216") > 0

    attr = compute_comb_norm(d["s2g_216"], d["g2s_216"])
    cmd.hide("everything", "pep216")
    cmd.show("cartoon", "pep216")
    apply_attr_bfactor_and_color("pep216", attr)

    style_key_sticks("pep216", [6, 7, 9, 12, 17, 18, 19])

    cmd.show("surface", "pep216")
    cmd.set("surface_color", "surf_tint", "pep216")
    cmd.set("transparency", 0.78, "pep216")

    cmd.orient("pep216")
    cmd.turn("y", 14)
    cmd.turn("x", 12)
    cmd.zoom("pep216", buffer=3.2)

    W, H = 1600, 1120
    record_residue_2d_coords("B", "pep216", W, H)
    cmd.save(os.path.join(OUT_DIR, "Test_216_NeuroEGNN_attr.pdb"), "pep216")
    cmd.save(os.path.join(OUT_DIR, "panelB_test216.pse"))
    cmd.ray(W, H)
    cmd.png(os.path.join(OUT_DIR, "panelB_test216.png"), dpi=300)
    print("Rendered panelB_test216.png")


def render_panel_C():
    cmd.reinitialize()
    setup_render_style()
    cmd.load(os.path.join(PDB_DIR, "test_651.pdb"), "pep651")
    assert cmd.count_atoms("pep651") > 0

    attr = compute_comb_norm(d["s2g_651"], d["g2s_651"])
    cmd.hide("everything", "pep651")
    cmd.show("cartoon", "pep651")
    apply_attr_bfactor_and_color("pep651", attr)

    style_key_sticks("pep651", [12, 13, 14, 16, 17, 18])

    cmd.show("surface", "pep651 and resi 12-18")
    cmd.set("transparency", 0.68, "pep651")

    cmd.orient("pep651")
    cmd.turn("z", -12)
    cmd.turn("x", 15)
    cmd.zoom("pep651", buffer=3.0)

    W, H = 1600, 1120
    record_residue_2d_coords("C", "pep651", W, H)
    cmd.save(os.path.join(OUT_DIR, "Test_651_NeuroEGNN_attr.pdb"), "pep651")
    cmd.save(os.path.join(OUT_DIR, "panelC_test651.pse"))
    cmd.ray(W, H)
    cmd.png(os.path.join(OUT_DIR, "panelC_test651.png"), dpi=300)
    print("Rendered panelC_test651.png")


def render_panel_D():
    attr_before = compute_comb_norm(d["s2g_156"], d["g2s_156"])
    attr_after_sub = compute_comb_norm(d["s2g_cf_156"][1:, 1:], d["g2s_cf_156"][1:, 1:])
    attr_after = np.concatenate([[0.45], attr_after_sub])

    W, H = 780, 1120
    for tag, attr_vec, masked in [("D_before", attr_before, False), ("D_after", attr_after, True)]:
        cmd.reinitialize()
        setup_render_style()
        cmd.load(os.path.join(PDB_DIR, "test_156.pdb"), "pep156")
        assert cmd.count_atoms("pep156") > 0

        cmd.hide("everything", "pep156")
        cmd.show("cartoon", "pep156")
        apply_attr_bfactor_and_color("pep156", attr_vec)
        style_key_sticks("pep156", [1, 4, 6, 7, 9, 10])

        if masked:
            cmd.color("ghost_gray", "pep156 and resi 1")
            cmd.set("cartoon_transparency", 0.65, "pep156 and resi 1")
            cmd.set("stick_transparency", 0.65, "pep156 and resi 1")

        cmd.show("surface", "pep156")
        cmd.set("surface_color", "surf_tint", "pep156")
        cmd.set("transparency", 0.78, "pep156")

        cmd.orient("pep156")
        cmd.turn("z", 44)
        cmd.turn("x", -14)
        cmd.zoom("pep156", buffer=4.4)

        record_residue_2d_coords(tag, "pep156", W, H)
        cmd.save(os.path.join(OUT_DIR, f"{tag}_test156.pse"))
        cmd.ray(W, H)
        cmd.png(os.path.join(OUT_DIR, f"{tag}_test156.png"), dpi=300)
        print(f"Rendered {tag}_test156.png")


if __name__ == "__main__":
    render_panel_A()
    render_panel_B()
    render_panel_C()
    render_panel_D()
    with open(os.path.join(OUT_DIR, "pymol_views.json"), "w") as f:
        json.dump(VIEWS_META, f, indent=2)
    cmd.quit()
