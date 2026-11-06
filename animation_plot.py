import glob
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------- reading ---
def read_dataitem(item, base_dir):
    """Read a <DataItem> of the form 'file.h5:/path/in/file'."""
    fname, path = item.text.strip().split(":", 1)
    with h5py.File(os.path.join(base_dir, fname), "r") as h:
        return h[path][()]


def load_xdmf(xdmf_path):
    """
    Returns a list of frames sorted by time:
      {"time": float, "topology": (n_cells, 3) int, "geometry": (n_pts, 3), "fields": {name: array}}
    Geometry is taken from each time step if it has its own <Geometry>,
    otherwise from the first (reference) mesh (the xi:include case).
    """
    base_dir = os.path.dirname(os.path.abspath(xdmf_path))
    domain = ET.parse(xdmf_path).getroot().find("Domain")

    # Reference mesh: first top-level Uniform grid
    ref = next(g for g in domain.findall("Grid") if g.get("GridType") == "Uniform")
    ref_topo = read_dataitem(ref.find("Topology/DataItem"), base_dir)
    ref_geom = read_dataitem(ref.find("Geometry/DataItem"), base_dir)

    frames = {}
    for coll in domain.findall("Grid"):
        if coll.get("GridType") != "Collection":
            continue
        for g in coll.findall("Grid"):
            t = float(g.find("Time").get("Value"))
            fr = frames.setdefault(
                round(t, 12),
                {"time": t, "topology": ref_topo, "geometry": ref_geom, "fields": {}},
            )
            # Moving mesh: a step can carry its own geometry/topology
            if g.find("Geometry/DataItem") is not None:
                fr["geometry"] = read_dataitem(g.find("Geometry/DataItem"), base_dir)
            if g.find("Topology/DataItem") is not None:
                fr["topology"] = read_dataitem(g.find("Topology/DataItem"), base_dir)
            for att in g.findall("Attribute"):
                fr["fields"][att.get("Name")] = read_dataitem(att.find("DataItem"), base_dir)

    return [frames[k] for k in sorted(frames)]


def load_series(pattern):
    """Load one or many .xdmf files (e.g. 'out/*.xdmf') into one time-sorted list."""
    frames = []
    for f in sorted(glob.glob(pattern)):
        frames += load_xdmf(f)
    return sorted(frames, key=lambda fr: fr["time"])


# ---------------------------------------------------------------- pyvista ---
def to_pv_mesh(frame):
    topo = frame["topology"]
    n = topo.shape[0]
    cells = np.hstack([np.full((n, 1), 3), topo]).ravel()           # [3, a, b, c, ...]
    celltypes = np.full(n, pv.CellType.TRIANGLE, dtype=np.uint8)
    pts = np.asarray(frame["geometry"], dtype=float)
    if pts.shape[1] == 2:
        pts = np.c_[pts, np.zeros(len(pts))]
    return pv.UnstructuredGrid(cells, celltypes, pts)


def velocity_magnitude(frame, name="f"):
    v = np.asarray(frame["fields"][name])
    return np.linalg.norm(v, axis=1)


def animate(frames, vel_name="f", out_file=None, fps=10):
    mesh = to_pv_mesh(frames[0])
    mesh["|velocity|"] = velocity_magnitude(frames[0], vel_name)

    # Fixed color range over the whole animation
    vmin = min(velocity_magnitude(f, vel_name).min() for f in frames)
    vmax = max(velocity_magnitude(f, vel_name).max() for f in frames)

    pl = pv.Plotter(off_screen=out_file is not None)
    pl.add_mesh(mesh, scalars="|velocity|", clim=[vmin, vmax], cmap="turbo",
                show_edges=False, scalar_bar_args={"title": "|velocity|"})
    pl.view_xy()
    text = pl.add_text(f"t = {frames[0]['time']:.5f}", font_size=10)

    def update(fr):
        mesh.points = np.asarray(fr["geometry"], dtype=float)        # mesh moves
        mesh["|velocity|"] = velocity_magnitude(fr, vel_name)        # values change
        text.SetText(2, f"t = {fr['time']:.5f}")

    if out_file:                                    # save to .mp4 / .gif
        if out_file.endswith(".gif"):
            pl.open_gif(out_file, fps=fps)
        else:
            pl.open_movie(out_file, framerate=fps)
        for fr in frames:
            update(fr)
            pl.write_frame()
        pl.close()
    else:                                           # interactive
        pl.show(interactive_update=True)
        while True:
            for fr in frames:
                update(fr)
                pl.update()
                pl.render()
                if pl._closed:
                    return


if __name__ == "__main__":
    # Single file, or a pattern such as "results/solution_*.xdmf"
    frames = load_series("runs/2026-11-06_11-16-10_alpha40_phi100_Re10000_res0.001_dt0.025_L3.0/solution_*.xdmf")
    print(f"{len(frames)} time steps, t = {frames[0]['time']} ... {frames[-1]['time']}")
    print("fields:", list(frames[0]["fields"]))

    animate(frames, vel_name="f")                           # interactive window
    # animate(frames, vel_name="f", out_file="anim.mp4")    # or save a movie