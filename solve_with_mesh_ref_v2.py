import os
import sys
import json
import copy
from datetime import datetime

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import refine, transfer_meshtag, RefinementOption
import ufl
from ufl import nabla_grad, Identity, grad, inner, dot, div

sys.path.append("/home/ferra/wing_dynamics/modules")
import plot as pltmd
from mesh_generation import generate_mesh_circular
from estrazione_spline import crea_spline_estesa, crea_funzioni
from custom_fun import custom_sin, zanzara, custom_di


# =====================================================================
# CONFIGURAZIONE (meglio modificare config.json che questi default)
# =====================================================================
DEFAULTS = {
    # L raggio dominio, r_x semilunghezza ala, r_y spessore ala,
    # N2 punti bordo esterno, resolution risoluzione al bordo ala
    "geometry": {"L": 6.0, "r_x": 0.5, "r_y": 0.02, "N2": 100, "resolution": 0.01},
    "flow": {"Re": 100},
    # T tempo finale, t_start tempo iniziale, dt_val_cost passo massimo
    "time": {"T": 0.75, "t_start": 0.0, "dt_val_cost": 0.07},
    "trajectory": {
        "alpha_deg": 40, "amp_phi_deg": 100,
        "period_phi": 0.5, "period_alpha": 0.5, "phase_alpha": 0.72,
        "sigma_phi": None, "sigma_alpha": None,
        "n_grid": 4096, "num_steps": 80,
    },
    "flags": {"flag_plot": True, "flag_plot_q": True, "start_from_previous": False},
    "output": {
        "base_dir": "runs",
        "save_times": [1.0, 2.0, 3.0, 4.0],
        "write_fields": True,    # salva mesh + campi a ogni passo (XDMF)
        "write_every": 1,        # ogni quanti passi scrivere
    },
    "refinement": {
        "enabled": True,
        "every": 5,              # valuta il raffinamento ogni N passi
        "theta": 0.5,            # raffina le celle con indicatore > theta * max
        "h_min": None,           # None -> resolution / 2
        "max_levels": 3,
        "max_cells": 200000,
    },
    "solver": {
        "type": "lu",            # "lu" (MUMPS) oppure "fieldsplit" (FGMRES + Schur + AMG)
        "ksp_rtol": 1e-6,
        "pin_pressure": True,
    },
}


def load_config(path):
    """Legge il JSON e lo fonde (sezione per sezione) con i default."""
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as fh:
        user = json.load(fh)
    for section, values in user.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def make_output_dir(cfg, comm):
    """Crea <base_dir>/<data_ora>_<parametri> e ci entra (chdir)."""
    g, fl, tr, tm = cfg["geometry"], cfg["flow"], cfg["trajectory"], cfg["time"]
    name = None
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = (f"{stamp}_alpha{tr['alpha_deg']}_phi{tr['amp_phi_deg']}"
                f"_Re{fl['Re']}_res{g['resolution']}_dt{tm['dt_val_cost']}_L{g['L']}")
    name = comm.bcast(name, root=0)
    out_dir = os.path.abspath(os.path.join(cfg["output"]["base_dir"], name))
    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config_used.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)
    comm.Barrier()
    os.chdir(out_dir)
    return out_dir


# =====================================================================
# FUNZIONI PER FEM
# =====================================================================
def NS_function_space(mesh, gdim):
    cell = mesh.basix_cell()
    P2 = element("Lagrange", cell, 2, shape=(gdim,))
    P1 = element("Lagrange", cell, 1)
    return fem.functionspace(mesh, mixed_element([P2, P1]))


def dofs_from_tags_NS(tags, W, mt, fdim):
    V0_x, _ = W.sub(0).sub(0).collapse()
    V0_y, _ = W.sub(0).sub(1).collapse()
    dofs = []
    for tag in tags:
        facets = mt.find(tag)
        dofs.append([
            fem.locate_dofs_topological((W.sub(0).sub(0), V0_x), fdim, facets),
            fem.locate_dofs_topological((W.sub(0).sub(1), V0_y), fdim, facets),
        ])
    return dofs


def define_form_NS(Re_inv, u, p, v, q, f, u_n, u_mesh, dt):
    return (
        inner((u - u_n) / dt, v) * ufl.dx                                     # transitorio
        + inner(dot(u - u_mesh, nabla_grad(u)), v) * ufl.dx                   # convezione (ALE)
        + Re_inv * inner(nabla_grad(u) + nabla_grad(u).T, grad(v)) * ufl.dx   # viscosita'
        - div(v) * p * ufl.dx                                                 # pressione
        + q * div(u) * ufl.dx                                                 # incomprimibilita'
        - inner(f, v) * ufl.dx                                                # forzante
    )


def eval_dt(f1_t, f2_t, dt_val_cost, t):
    v_max = max(np.max(np.abs(f1_t(t))), np.max(np.abs(f2_t(t))), 1)
    return dt_val_cost / (1 + v_max)


def compute_force_on_wing(u_h, p_h, mesh, mt, wing_tag, gdim, Re_inv, F_t):
    I = Identity(gdim)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    j = ufl.indices(1)[0]
    sigma = -p_h * I + Re_inv * (nabla_grad(u_h) + nabla_grad(u_h).T)
    Fx = -np.array([
        fem.assemble_scalar(fem.form(sigma[i, j] * n[j] * ds(wing_tag)))
        for i in range(gdim)
    ])
    if mesh.comm.rank == 0:
        print("  forza sul corpo interno:", Fx)
        F_t.append(Fx)
    return Fx


def compute_torque_on_wing(u_h, p_h, mesh, mt, wing_tag, gdim, Re_inv, M_t, x0):
    I = Identity(gdim)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    sigma = -p_h * I + Re_inv * (nabla_grad(u_h) + nabla_grad(u_h).T)
    r = ufl.SpatialCoordinate(mesh) - x0
    M = -fem.assemble_scalar(fem.form(
        (r[0] * dot(sigma[1, :], n) - r[1] * dot(sigma[0, :], n)) * ds(wing_tag)
    ))
    if mesh.comm.rank == 0:
        print("  Torque sul corpo interno:", M)
        M_t.append(M)
    return M


# =====================================================================
# SOLVER / PRECONDITIONER
# =====================================================================
def build_petsc_options(solver_cfg):
    base = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-7,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_linesearch_type": "bt",
        "snes_linesearch_damping": 0.8,
    }
    if solver_cfg["type"] == "lu":
        base.update({
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        })
    elif solver_cfg["type"] == "fieldsplit":
        base.update({
            "ksp_type": "fgmres",
            "ksp_rtol": solver_cfg["ksp_rtol"],
            "ksp_max_it": 200,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "upper",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_u_ksp_type": "preonly",
            "fieldsplit_u_pc_type": "hypre",
            "fieldsplit_u_pc_hypre_type": "boomeramg",
            "fieldsplit_p_ksp_type": "preonly",
            "fieldsplit_p_pc_type": "hypre",
            "fieldsplit_p_pc_hypre_type": "boomeramg",
        })
        # se hypre non e' installato: usa "gamg" e togli le chiavi *_hypre_type
    else:
        raise ValueError(f"solver.type sconosciuto: {solver_cfg['type']}")
    return base


def make_field_IS(W, parent_dofs):
    """IS (numerazione globale, solo DOF posseduti) per un sottospazio collassato di W."""
    imap = W.dofmap.index_map
    n_loc, off = imap.size_local, imap.local_range[0]
    d = np.asarray(parent_dofs)
    d = d[d < n_loc] + off
    return PETSc.IS().createGeneral(d.astype(PETSc.IntType), comm=W.mesh.comm)


# =====================================================================
# SALVATAGGIO TRAIETTORIA (mesh + campi)
# =====================================================================
class TrajectoryWriter:
    """
    Un file XDMF/HDF5 per segmento di mesh. La mesh e' scritta UNA volta
    (configurazione a inizio segmento); a ogni passo si scrivono:
      velocity, pressure   (interpolate su P1)
      displacement         (posizione corrente - posizione a inizio segmento)
    In ParaView: Warp By Vector su 'displacement', scale factor 1.
    """

    def __init__(self, mesh, filename):
        self.mesh = mesh
        self.gdim = mesh.geometry.dim
        self.filename = filename
        V1 = fem.functionspace(mesh, ("Lagrange", 1, (self.gdim,)))
        Q1 = fem.functionspace(mesh, ("Lagrange", 1))
        self.u = fem.Function(V1, name="velocity")
        self.p = fem.Function(Q1, name="pressure")
        self.d = fem.Function(V1, name="displacement")
        self.x_ref = mesh.geometry.x.copy()
        self.nodes_V = V1.dofmap.list.reshape(-1)
        self.nodes_G = mesh.geometry.dofmap.reshape(-1)
        self.times = []
        self.file = XDMFFile(mesh.comm, filename, "w")
        self.file.write_mesh(mesh)

    def write(self, t, u, p):
        """u, p: funzioni collassate. Chiamare DOPO il movimento della mesh."""
        self.u.interpolate(u)
        self.p.interpolate(p)
        d = self.d.x.array.reshape(-1, self.gdim)
        x = self.mesh.geometry.x
        d[self.nodes_V] = (x[self.nodes_G, :self.gdim]
                           - self.x_ref[self.nodes_G, :self.gdim])
        self.d.x.scatter_forward()
        for f in (self.u, self.p, self.d):
            self.file.write_function(f, t)
        self.times.append(float(t))

    def close(self):
        self.file.close()
        return {"file": self.filename,
                "t_start": self.times[0] if self.times else None,
                "t_end": self.times[-1] if self.times else None,
                "n_frames": len(self.times)}


# =====================================================================
# STATO FEM (tutto cio' che dipende dalla mesh)
# =====================================================================
class FlowState:
    """Raccoglie mesh, tag, spazi, funzioni e BC-dofs. Si ricrea a ogni raffinamento."""

    def __init__(self, mesh, mt, Re, dt_val):
        self.mesh, self.mt = mesh, mt
        self.gdim = mesh.geometry.dim
        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        mesh.topology.create_connectivity(self.fdim, self.tdim)

        self.W = NS_function_space(mesh, self.gdim)
        self.w = fem.Function(self.W)       # n+1
        self.w_n = fem.Function(self.W)     # n
        self.u, self.p = ufl.split(self.w)
        self.u_n, _ = ufl.split(self.w_n)
        self.v, self.q = ufl.TestFunctions(self.W)

        self.Re_inv = fem.Constant(mesh, PETSc.ScalarType(1 / Re))
        self.dt = fem.Constant(mesh, PETSc.ScalarType(dt_val))

        self.dofs = dofs_from_tags_NS([2, 3], self.W, mt, self.fdim)  # 2 = esterno, 3 = ala

        self.V0, self.map_u = self.W.sub(0).collapse()
        self.Q0, self.map_p = self.W.sub(1).collapse()
        self.is_u = make_field_IS(self.W, self.map_u)
        self.is_p = make_field_IS(self.W, self.map_p)
        V0_x, _ = self.W.sub(0).sub(0).collapse()
        V0_y, _ = self.W.sub(0).sub(1).collapse()
        self.u_mesh = fem.Function(self.V0)
        self.u_flow = fem.Function(V0_x)
        self.u_wing_x = fem.Function(V0_x)
        self.u_wing_y = fem.Function(V0_y)

        P1v = fem.functionspace(mesh, ("Lagrange", 1, (self.gdim,)))
        self.u_out_p1 = fem.Function(P1v)
        self.u_mesh_P1 = fem.Function(P1v)
        # mappa nodi P1 <-> nodi della geometria (l'ordine non e' garantito uguale)
        self.nodes_V = P1v.dofmap.list.reshape(-1)
        self.nodes_G = mesh.geometry.dofmap.reshape(-1)

        # pin pressione: un solo DOF sul bordo esterno (tag 2), solo sul rank 0
        pin = fem.locate_dofs_topological(self.W.sub(1), self.fdim, mt.find(2))
        self.pin_dofs = pin[:1] if (mesh.comm.rank == 0 and len(pin) > 0) else None

    def initial_bcs(self):
        d, W = self.dofs, self.W
        return [
            fem.dirichletbc(PETSc.ScalarType(0.0), d[0][0][0], W.sub(0).sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0.0), d[0][1][0], W.sub(0).sub(1)),
            fem.dirichletbc(PETSc.ScalarType(0.0), d[1][0][0], W.sub(0).sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0.0), d[1][1][0], W.sub(0).sub(1)),
        ]

    def move_mesh(self, dt_val):
        """
        Sposta la mesh di u_mesh*dt rispettando la mappa nodi P1 -> geometria.
        Ritorna lo spostamento nell'ordine dei DOF P1 (per le animazioni).
        """
        self.u_mesh_P1.interpolate(self.u_mesh)
        disp = self.u_mesh_P1.x.array.reshape(-1, self.gdim) * dt_val
        inc = np.zeros((self.mesh.geometry.x.shape[0], self.gdim))
        inc[self.nodes_G] = disp[self.nodes_V]
        self.mesh.geometry.x[:, :self.gdim] += inc
        return disp


# =====================================================================
# RAFFINAMENTO DINAMICO
# =====================================================================
def compute_indicator(S):
    """Indicatore per cella eta_K = h_K * |grad u|. Ritorna (eta, h) sulle celle locali."""
    mesh = S.mesh
    u_c = S.w.sub(0).collapse()
    DG0 = fem.functionspace(mesh, ("DG", 0))
    h_expr = ufl.CellDiameter(mesh)

    eta_f, h_f = fem.Function(DG0), fem.Function(DG0)
    pts = DG0.element.interpolation_points
    eta_f.interpolate(fem.Expression(h_expr * ufl.sqrt(inner(grad(u_c), grad(u_c))), pts))
    h_f.interpolate(fem.Expression(h_expr, pts))

    n_loc = mesh.topology.index_map(S.tdim).size_local
    return eta_f.x.array[:n_loc].copy(), h_f.x.array[:n_loc].copy()


def mark_edges(S, ref_cfg, h_min):
    """Ritorna gli edge da raffinare (array int32) oppure None se non c'e' nulla da fare."""
    mesh = S.mesh
    comm = mesh.comm
    eta, h = compute_indicator(S)

    n_total = comm.allreduce(len(eta), op=MPI.SUM)
    eta_max = comm.allreduce(eta.max() if len(eta) else 0.0, op=MPI.MAX)
    if eta_max <= 0.0:
        return None

    marked = np.flatnonzero((eta > ref_cfg["theta"] * eta_max) & (h > h_min)).astype(np.int32)

    budget = ref_cfg["max_cells"] - n_total
    if budget <= 0:
        if comm.rank == 0:
            print(f"  [refine] max_cells raggiunto ({n_total}), niente raffinamento")
        return None
    k_loc = max(budget // 4 // comm.size, 0)
    if len(marked) > k_loc:
        marked = marked[np.argsort(eta[marked])[::-1][:k_loc]]

    n_marked = comm.allreduce(len(marked), op=MPI.SUM)
    if n_marked == 0:
        return None
    if comm.rank == 0:
        print(f"  [refine] celle marcate: {n_marked}/{n_total}")

    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(S.tdim, 1)
    c2e = mesh.topology.connectivity(S.tdim, 1)
    if len(marked) == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate([c2e.links(c) for c in marked])).astype(np.int32)


def transfer_fields(S_old, S_new):
    """Interpola w e w_n dalla mesh vecchia alla nuova (mesh annidate: P2/P1 esatti)."""
    imap = S_new.mesh.topology.index_map(S_new.tdim)
    cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    data_u = fem.create_interpolation_data(S_new.V0, S_old.V0, cells, padding=1e-10)
    data_p = fem.create_interpolation_data(S_new.Q0, S_old.Q0, cells, padding=1e-10)

    for src, dst in ((S_old.w, S_new.w), (S_old.w_n, S_new.w_n)):
        u_new = fem.Function(S_new.V0)
        p_new = fem.Function(S_new.Q0)
        u_new.interpolate_nonmatching(src.sub(0).collapse(), cells, interpolation_data=data_u)
        p_new.interpolate_nonmatching(src.sub(1).collapse(), cells, interpolation_data=data_p)
        dst.sub(0).interpolate(u_new)
        dst.sub(1).interpolate(p_new)
        dst.x.scatter_forward()


def refine_state(S, edges, Re, dt_val):
    """Raffina la mesh negli edge dati, trasferisce tag e soluzione. Ritorna il nuovo FlowState."""
    mesh_old = S.mesh
    mesh_old.topology.create_connectivity(S.fdim, S.tdim)

    mesh_new, parent_cell, parent_facet = refine(
        mesh_old, edges, option=RefinementOption.parent_cell_and_facet)

    mesh_new.topology.create_entities(S.fdim)
    mesh_new.topology.create_connectivity(S.fdim, S.tdim)
    mt_new = transfer_meshtag(S.mt, mesh_new, parent_cell, parent_facet)

    S_new = FlowState(mesh_new, mt_new, Re, dt_val)
    transfer_fields(S, S_new)
    return S_new


# =====================================================================
# MAIN
# =====================================================================
def main():
    config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    cfg = load_config(config_path)

    # --- parametri
    g, tm, tr, fl = cfg["geometry"], cfg["time"], cfg["trajectory"], cfg["flags"]
    ref, out, solver_cfg = cfg["refinement"], cfg["output"], cfg["solver"]
    L, r_x, r_y, N2, resolution = g["L"], g["r_x"], g["r_y"], g["N2"], g["resolution"]
    Re = cfg["flow"]["Re"]
    T, t, dt_val_cost = tm["T"], tm["t_start"], tm["dt_val_cost"]
    alpha_deg, amp_phi_deg = tr["alpha_deg"], tr["amp_phi_deg"]
    alpha_amp = np.deg2rad(alpha_deg)
    amp_phi = np.deg2rad(amp_phi_deg)
    mesh_angle = -(np.pi / 2 - alpha_amp)
    c_r = (r_x, 0)
    N = int(4 * r_x / resolution)
    flag_plot, flag_plot_q = fl["flag_plot"], fl["flag_plot_q"]
    save_times = out["save_times"]
    h_min = ref["h_min"] if ref["h_min"] is not None else 0.5 * resolution

    # --- cartella di output
    out_dir = make_output_dir(cfg, MPI.COMM_WORLD)
    print("Output in:", out_dir)

    # --- mesh e traiettoria
    generate_mesh_circular(L, r_x, r_y, N, N2, mesh_angle, c_r, resolution)

    f1_t, f2_t, f3_t, x = custom_di(
        T, num_steps=tr["num_steps"],
        period_phi=tr["period_phi"], amp_phi=amp_phi,
        period_alpha=tr["period_alpha"], amp_alpha=alpha_amp,
        sigma_phi=tr["sigma_phi"], sigma_alpha=tr["sigma_alpha"],
        phase_alpha=tr["phase_alpha"], n_grid=tr["n_grid"],
    )  # f1_t = dphi/dt, f2_t = dalpha/dt, f3_t = d2phi/dt2

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with XDMFFile(mesh.comm, "mf.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, name="Grid")

    # --- stato FEM iniziale
    dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
    S = FlowState(mesh, mt, Re, dt_val)
    gdim = S.gdim

    if fl["start_from_previous"]:   # NON implementato/testato
        data = np.load("state_t1.npz")
        u_prev, p_prev = S.w_n.split()
        u_prev.x.array[:] = data["u"]
        p_prev.x.array[:] = data["p"]
        S.w_n.x.scatter_forward()
        S.w.x.array[:] = S.w_n.x.array
        S.w.x.scatter_forward()
    else:
        bcs0 = S.initial_bcs()
        S.w.x.array[:] = 0.0
        S.w_n.x.array[:] = 0.0
        fem.set_bc(S.w.x.array, bcs0)
        fem.set_bc(S.w_n.x.array, bcs0)

    petsc_options = build_petsc_options(solver_cfg)

    # --- animazioni (ricreate a ogni raffinamento, una gif per segmento)
    anim = {"plotter": None, "update": None, "plotter_q": None, "update_q": None}

    def make_plotters(S, seg):
        u_out = S.w.sub(0).collapse()
        if flag_plot:
            anim["plotter"], anim["update"] = pltmd.create_elasticity_animation(
                S.W.sub(0), S.mesh, u_out, S.gdim, fem,
                filename=f"elastic_block_seg{seg}.gif",
                show_edges=False, traslation=True, velocity=0)
        if flag_plot_q:
            anim["plotter_q"], anim["update_q"] = pltmd.create_qcriterion_animation(
                S.mesh, u_out, fem, filename=f"q_criterion_seg{seg}.gif",
                q_threshold=10, traslation=True, above_color="yellow",
                below_color="green", show_mesh_outline=False, zoom=4.0)

    def close_plotters():
        for key in ("plotter", "plotter_q"):
            if anim[key] is not None:
                anim[key].close()
                anim[key] = None

    seg = 0
    make_plotters(S, seg)

    # --- writer della traiettoria (mesh + campi)
    segments_info = []
    writer = TrajectoryWriter(S.mesh, f"fields_seg{seg}.xdmf") if out["write_fields"] else None

    # --- cinematica corrente
    kin = {"omega": 0.0, "v_p": 0.0, "a_p": 0.0, "c_p": np.array([0.0, 0.0, 0.0])}

    def rotational_field_x(xx):
        return -kin["omega"] * (xx[1] - kin["c_p"][1])

    def rotational_field_y(xx):
        return kin["omega"] * (xx[0] - kin["c_p"][0])

    def rotational_field(xx):
        return np.vstack((
            -kin["omega"] * (xx[1] - kin["c_p"][1]),
            kin["omega"] * (xx[0] - kin["c_p"][0]),
            np.zeros_like(xx[2]),
        ))

    def step_solve(S, step, t):
        """Un passo temporale. Ritorna True se Newton e' convergito."""
        S.u_mesh.interpolate(rotational_field)
        f = fem.Constant(S.mesh, PETSc.ScalarType((kin["a_p"],) + (0.0,) * (gdim - 1)))

        S.u_flow.x.array[:] = -kin["v_p"]
        S.u_wing_x.interpolate(rotational_field_x)
        S.u_wing_y.interpolate(rotational_field_y)
        d, W = S.dofs, S.W
        bcs = [
            fem.dirichletbc(S.u_flow, d[0][0], W.sub(0).sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0.0), d[0][1][0], W.sub(0).sub(1)),
            fem.dirichletbc(S.u_wing_x, d[1][0], W.sub(0).sub(0)),
            fem.dirichletbc(S.u_wing_y, d[1][1], W.sub(0).sub(1)),
        ]
        if solver_cfg["pin_pressure"] and S.pin_dofs is not None:
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), S.pin_dofs, W.sub(1)))

        F = define_form_NS(S.Re_inv, S.u, S.p, S.v, S.q, f, S.u_n, S.u_mesh, S.dt)
        problem = NonlinearProblem(F, S.w, bcs=bcs,
                                   petsc_options_prefix="navier_stokes_",
                                   petsc_options=petsc_options)
        if solver_cfg["type"] == "fieldsplit":
            pc = problem.solver.getKSP().getPC()
            pc.setFieldSplitIS(("u", S.is_u), ("p", S.is_p))
        problem.solve()
        S.u_mesh.x.scatter_forward()

        reason = problem.solver.getConvergedReason()
        n_its = problem.solver.getIterationNumber()
        p_arr = S.w.sub(1).collapse().x.array
        if S.mesh.comm.rank == 0:
            print(f"step {step}, t = {t:.4f}, Newton its = {n_its}, reason = {reason}")
            print(f"max|w| = {np.max(np.abs(S.w.x.array)):.4e}, "
                  f"NaN: {np.isnan(S.w.x.array).any()}")
            print(f"p_mean={p_arr.mean():.6e}, p_min={p_arr.min():.6e}, "
                  f"p_max={p_arr.max():.6e}, range={p_arr.max() - p_arr.min():.6e}")
        return reason > 0

    # --- ciclo temporale
    F_t, M_t, v_t, T_t, alpha_t, beta_t = [], [], [], [], [], []
    snap = {"X": [], "t": []}     # snapshot del segmento corrente (per POD)
    saved_states = []
    alpha_pos, beta_pos = 0.0, 0.0
    n_refinements = 0
    step = 0

    while t < T:
        step += 1
        t += dt_val
        kin["v_p"] = -f1_t(t)[0]
        kin["omega"] = f2_t(t)[0]
        kin["a_p"] = -f3_t(t)[0]
        print(f"step {step}, t = {t:.4f}, v_p = {kin['v_p']:.4f}, "
              f"omega = {kin['omega']:.4f}, a_p = {kin['a_p']:.4f}")

        if not step_solve(S, step, t):
            print(f"!! SNES non converso al passo {step}: interrompo e salvo i risultati parziali")
            break

        u_sol = S.w.sub(0).collapse()
        p_sol = S.w.sub(1).collapse()
        S.u_out_p1.interpolate(u_sol)
        snap["X"].append(S.u_out_p1.x.array.copy())
        snap["t"].append(t)

        # salvataggio stati a tempi prefissati
        tol = dt_val / 2
        for ts in save_times:
            if ts not in saved_states and abs(t - ts) < tol:
                np.savez(f"state_t{ts:.0f}.npz",
                         u_coords=S.V0.tabulate_dof_coordinates(), u=u_sol.x.array.copy(),
                         p_coords=S.Q0.tabulate_dof_coordinates(), p=p_sol.x.array.copy())
                saved_states.append(ts)

        # movimento mesh
        disp = S.move_mesh(dt_val)

        # salvataggio traiettoria (dopo il movimento della mesh)
        if writer is not None and step % out["write_every"] == 0:
            writer.write(t, u_sol, p_sol)

        # animazioni
        try:
            if flag_plot:
                anim["update"](u_sol, disp, kin["v_p"], dt_val)
            if flag_plot_q:
                anim["update_q"](u_sol, disp, kin["v_p"], dt_val)
        except Exception as e:
            print(f"Errore animazione al passo {step}: {e}")

        S.w_n.x.array[:] = S.w.x.array

        # forze e momenti
        compute_force_on_wing(u_sol, p_sol, S.mesh, S.mt, 3, gdim, S.Re_inv, F_t)
        compute_torque_on_wing(u_sol, p_sol, S.mesh, S.mt, 3, gdim, S.Re_inv, M_t, kin["c_p"])

        v_t.append(kin["v_p"])
        T_t.append(t)
        alpha_pos += kin["v_p"] * dt_val
        beta_pos += kin["omega"] * dt_val
        alpha_t.append(alpha_pos)
        beta_t.append(beta_pos)

        dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
        S.dt.value = dt_val

        # ---------------- raffinamento dinamico ----------------
        if (ref["enabled"] and step % ref["every"] == 0
                and n_refinements < ref["max_levels"] and t < T):
            edges = mark_edges(S, ref, h_min)
            if edges is not None:
                if S.mesh.comm.rank == 0:
                    np.savez(f"snapshots_seg{seg}.npz", X=np.array(snap["X"]),
                             t=np.array(snap["t"]), coords=S.mesh.geometry.x.copy())
                snap = {"X": [], "t": []}

                close_plotters()
                if writer is not None:
                    segments_info.append(writer.close())

                S = refine_state(S, edges, Re, dt_val)
                n_refinements += 1
                seg += 1
                make_plotters(S, seg)
                if out["write_fields"]:
                    writer = TrajectoryWriter(S.mesh, f"fields_seg{seg}.xdmf")

                n_cells = S.mesh.comm.allreduce(
                    S.mesh.topology.index_map(S.tdim).size_local, op=MPI.SUM)
                if S.mesh.comm.rank == 0:
                    print(f"  [refine] livello {n_refinements}: {n_cells} celle")

    # --- chiusura e post-processing
    close_plotters()
    if writer is not None:
        segments_info.append(writer.close())
        if MPI.COMM_WORLD.rank == 0:
            with open("segments.json", "w") as fh:
                json.dump(segments_info, fh, indent=2)

    T_t, v_t = np.array(T_t), np.array(v_t)
    F_t, M_t = np.array(F_t), np.array(M_t)
    alpha_t, beta_t = np.array(alpha_t), np.array(beta_t)

    pltmd.plot_force(T_t, F_t, v_t)

    data = np.column_stack((T_t, v_t, F_t, M_t, alpha_t, beta_t))
    header = ("time v_p force torque alpha beta\n"
              "config:\n" + json.dumps(cfg))
    np.savetxt("aerodynamic_history.txt", data, header=header, comments="# ")

    # POD solo sull'ultimo segmento (mesh costante); gli altri in snapshots_seg*.npz
    if len(snap["X"]) > 0:
        pltmd.save_POD(np.array(snap["X"]), S.mesh.geometry.x, np.array(snap["t"]))


if __name__ == "__main__":
    main()