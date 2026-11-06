import os
import sys
import json
import copy
from datetime import datetime

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt

from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
import ufl
from ufl import nabla_grad, Identity, grad, inner, dot, div

sys.path.append("/home/ferra/wing_dynamics/modules")
import plot as pltmd
from mesh_generation import generate_mesh_circular, generate_mesh_naca
from estrazione_spline import crea_spline_estesa, crea_funzioni
from custom_fun import custom_test


def load_config(path):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as fh:
        user = json.load(fh)
    for section, values in user.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def make_output_dir(cfg, comm):
    g, fl, tr, tm = cfg["geometry"], cfg["flow"], cfg["trajectory"], cfg["time"]
    name = None
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = (f"{stamp}_alpha{tr['alpha0']}_phi{tr['phi']}"
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


def MD_function_space(mesh, gdim):  # spazio per la deformazione della mesh
    cell = mesh.basix_cell()
    P2 = element("Lagrange", cell, 2, shape=(gdim,))
    return fem.functionspace(mesh, P2)


def define_form_NS(Re_inv, u, p, v, q, f, u_n, u_mesh, dt, ds, n, tag_outlet):
    """Navier-Stokes ALE nel riferimento traslante dell'ala.
    Il termine al bordo (do-nothing) e' applicato solo sull'outlet."""
    sigma = -p * Identity(len(u)) + Re_inv * (nabla_grad(u) + nabla_grad(u).T)
    return (
        inner((u - u_n) / dt, v) * ufl.dx                                    # transitorio
        + inner(dot(u - u_mesh, nabla_grad(u)), v) * ufl.dx                  # convezione ALE
        + Re_inv * inner(nabla_grad(u) + nabla_grad(u).T, grad(v)) * ufl.dx  # viscosita'
        - div(v) * p * ufl.dx                                                # pressione
        + q * div(u) * ufl.dx                                                # incomprimibilita'
        - inner(f, v) * ufl.dx                                               # forzante (forza apparente)
        - dot(dot(sigma, n), v) * ds(tag_outlet)                             # outlet
    )


def define_form_mesh_deformation(u_mesh, v_mesh, gdim, mesh):
    """Problema di Laplace lineare per il campo armonico di velocita' della mesh."""
    a = inner(grad(u_mesh), grad(v_mesh)) * ufl.dx
    L = inner(fem.Constant(mesh, PETSc.ScalarType((0.0,) * gdim)), v_mesh) * ufl.dx
    return a, L


def eval_dt(f1_t, f2_t, dt_val_cost, t):
    v_max = max(np.max(np.abs(f1_t(t))), np.max(np.abs(f2_t(t))), 1)
    return dt_val_cost / (1 + v_max)


def _pad3(vec, gdim):
    out = np.zeros(3)
    out[:gdim] = vec
    return out


def compute_force_on_wing(u_h, p_h, mesh, wing_tag, gdim, Re_inv, F_t, n, ds):
    """Forza del fluido sul profilo: F = -int sigma.n ds (n uscente dal fluido, quindi entrante nell'ala)."""
    visc = Re_inv * (grad(u_h) + grad(u_h).T)
    Fp = np.array([mesh.comm.allreduce(
        fem.assemble_scalar(fem.form((p_h * n[i]) * ds(wing_tag))), op=MPI.SUM)
        for i in range(gdim)])
    Fmu = np.array([mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(-dot(visc, n)[i] * ds(wing_tag))), op=MPI.SUM)
        for i in range(gdim)])
    Fp, Fmu = _pad3(Fp, gdim), _pad3(Fmu, gdim)
    F = Fp + Fmu
    F_t.append(F)
    if mesh.comm.rank == 0:
        print("  forza sull'ala:", F, " (pressione:", Fp, ", viscosita':", Fmu, ")")
    return F, Fp, Fmu


# Tag dei bordi della mesh
TAG_INLET, TAG_OUTLET, TAG_LATERAL, TAG_WING = 2, 3, 4, 5

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
DEFAULTS = {
    "geometry": {"L": 10.0, "H": 5.0, "a": 0.5, "chord": 1.0, "alpha": 3.1415, "N2": 200,
                 "c_r": [0.0, 0.0], "size_far": 0.3, "dist_max": 1.5,
                 "resolution": 0.001, "mesh_file": "wing.msh"},
    "flow": {"Re": 10000},
    "time": {"T": 3, "t_start": 0.0, "dt_val_cost": 0.05},
    "trajectory": {
        "phi": 40, "y0": 0.1,
        "alpha0": 5, "lamb": 10, "freq":1
    },
    "flags": {"flag_plot": True, "flag_plot_q": False, "start_from_previous": False},
    "output": {"base_dir": "runs", "save_times": [1.0, 2.0, 3.0, 4.0]}
}

# =====================================================================
# MAIN
# =====================================================================
def main():
    test = True   # True: caso test, False: simulazione normale
    if test:
        config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "config_test.json")
    else:
        return 2

    cfg = load_config(config_path)

    # --- parametri
    g, tm, tr, fl = cfg["geometry"], cfg["time"], cfg["trajectory"], cfg["flags"]
    L, H, chord, resolution, a_naca = g["L"], g["H"], g["chord"], g["resolution"], g["a"]
    Re = cfg["flow"]["Re"]
    T, t, dt_val_cost = tm["T"], tm["t_start"], tm["dt_val_cost"]
    alpha_amp = np.deg2rad(tr["alpha0"])
    amp_phi = np.deg2rad(tr["phi"])
    c_r = np.array(g["c_r"], dtype=float)         # centro di rotazione (fisso nel riferimento ala)
    flag_plot, flag_plot_q = fl["flag_plot"], fl["flag_plot_q"]
    save_times = cfg["output"]["save_times"]
    U = 1#(2*tr["y0"])/tr["St"]  # frequenza di oscillazione della traiettoria (Hz)
    f = tr["St"]/(2*0.75)
    omega_angular = 2 * np.pi * f
    T = 1 / f * T
    dt_val_cost = dt_val_cost * (1 / f)

    out_dir = make_output_dir(cfg, MPI.COMM_WORLD)
    print("Output in:", out_dir)

    # --- mesh e traiettoria
    generate_mesh_naca(L, H, a_naca, chord=chord, alpha=alpha_amp+np.pi, c_r=tuple(c_r),
                       t=0.12, n_pts=80,
                       size_wall=resolution, size_far=g["size_far"], dist_max=g["dist_max"],
                       mesh_file=g["mesh_file"])

    f1_t, f2_t, f3_t = custom_test(omega_angular, U, tr["y0"], alpha_amp, amp_phi, tr["lamb"], T)  # f1 = dphi/dt, f2 = dalpha/dt, f3 = d2phi/dt2

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    gdim = mesh.geometry.dim
    with XDMFFile(mesh.comm, "mf.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, name="Grid")

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    n = ufl.FacetNormal(mesh)

    # --- spazi e funzioni NS
    W = NS_function_space(mesh, gdim)
    w = fem.Function(W)
    w_n = fem.Function(W)
    (u, p) = ufl.split(w)
    (u_n, _) = ufl.split(w_n)
    (v, q) = ufl.TestFunctions(W)

    # --- spazi e funzioni per la deformazione della mesh
    W_mesh = MD_function_space(mesh, gdim)
    u_mesh = fem.Function(W_mesh)                 # velocita' mesh (soluzione di Laplace)
    u_mesh_trial = ufl.TrialFunction(W_mesh)
    v_mesh = ufl.TestFunction(W_mesh)

    dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
    Re_inv = fem.Constant(mesh, PETSc.ScalarType(1 / Re))
    dt = fem.Constant(mesh, PETSc.ScalarType(dt_val))
    f = fem.Constant(mesh, PETSc.ScalarType((0.0,) * gdim))   # forza apparente (traslazione)

    # --- cinematica corrente
    kin = {"omega": 0.0, "v_p": [0.0, 0.0], "a_p": [0.0, 0.0]}

    # --- campi di velocita' imposti al bordo (aggiornati in place ad ogni passo)
    V0, map_u = W.sub(0).collapse()
    Q0, map_p = W.sub(1).collapse()
    V0_y, _ = W.sub(0).sub(1).collapse()

    

    

    u_inlet = fem.Function(V0)        # (-v_p, 0) su inlet
    u_wing_ns = fem.Function(V0)      # velocita' rigida di rotazione su ala (fluido)
    u_wing_md = fem.Function(W_mesh)  # velocita' rigida di rotazione su ala (mesh)

    def rotational_field_and_traslation_along_y(x):
        """omega x (r - c) in 2D: (-omega (y-cy),  omega (x-cx))."""
        vals = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        vals[0] = -kin["omega"] * (x[1] - c_r[1])
        vals[1] = kin["omega"] * (x[0] - c_r[0]) + kin["v_p"][1]
        return vals

    def inlet_field(x):
        vals = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        vals[0] = -kin["v_p"][0]
        return vals

    facets = {tg: mt.find(tg) for tg in (TAG_INLET, TAG_OUTLET, TAG_LATERAL, TAG_WING)}

    # --- BC Navier-Stokes
    dofs_in = fem.locate_dofs_topological((W.sub(0), V0), fdim, facets[TAG_INLET])
    dofs_wing = fem.locate_dofs_topological((W.sub(0), V0), fdim, facets[TAG_WING])
    dofs_lat_y = fem.locate_dofs_topological(W.sub(0).sub(1), fdim, facets[TAG_LATERAL])
    dofs_lat_y_p = fem.locate_dofs_topological((W.sub(1), Q0), fdim, facets[TAG_OUTLET])  
    bcs_NS = [
        fem.dirichletbc(u_inlet, dofs_in, W.sub(0)),                              # corrente -v_p
        fem.dirichletbc(u_wing_ns, dofs_wing, W.sub(0)),                          # aderenza + rotazione
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_lat_y, W.sub(0).sub(1)),      # pareti laterali: u_y = 0
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_lat_y_p[0], W.sub(1))             # pareti laterali: p = 0

    ]

    # --- BC deformazione mesh: nulla sui bordi esterni, rotazione rigida sull'ala
    bcs_MD = []
    zero_md = fem.Function(W_mesh)
    for tg in (TAG_INLET, TAG_OUTLET, TAG_LATERAL):
        d = fem.locate_dofs_topological(W_mesh, fdim, facets[tg])
        bcs_MD.append(fem.dirichletbc(zero_md, d))
    d_wing_md = fem.locate_dofs_topological(W_mesh, fdim, facets[TAG_WING])
    bcs_MD.append(fem.dirichletbc(u_wing_md, d_wing_md))

    # --- condizione iniziale
    w.x.array[:] = 0.0
    w_n.x.array[:] = 0.0
    u_inlet.interpolate(inlet_field)
    fem.set_bc(w.x.array, bcs_NS)
    fem.set_bc(w_n.x.array, bcs_NS)

    # --- problema lineare per la deformazione (costruito una volta)
    a_md, L_md = define_form_mesh_deformation(u_mesh_trial, v_mesh, gdim, mesh)
    mesh_problem = LinearProblem(
        a_md, L_md, u=u_mesh, bcs=bcs_MD,
        petsc_options_prefix="mesh_def_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )


    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-7,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "snes_monitor": None,

        # Iterative solver for the Newton Jacobian
        "ksp_type": "gmres",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 200,
        "ksp_monitor": None,

        # Velocity-pressure block preconditioner
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_schur_precondition": "selfp",

        # Velocity block
        "fieldsplit_u_ksp_type": "preonly",
        "fieldsplit_u_pc_type": "gamg",

        # Pressure Schur-complement approximation
        "fieldsplit_p_ksp_type": "preonly",
        "fieldsplit_p_pc_type": "jacobi",

        "snes_linesearch_type": "bt",
        "snes_linesearch_damping": 0.8,
    }
        
    F_form = define_form_NS(Re_inv, u, p, v, q, f, u_n, u_mesh, dt, ds, n, TAG_OUTLET)
    ns_problem = NonlinearProblem(F_form, w, bcs=bcs_NS,
                                  petsc_options_prefix="navier_stokes_",
                                  petsc_options=petsc_options)
    
    is_u = PETSc.IS().createGeneral(
            np.asarray(map_u, dtype=PETSc.IntType),
            comm=mesh.comm
        )
    
    is_p = PETSc.IS().createGeneral(
        np.asarray(map_p, dtype=PETSc.IntType),
        comm=mesh.comm
    )

    ksp = ns_problem.solver.getKSP()
    pc = ksp.getPC()

    pc.setFieldSplitIS(
        ("u", is_u),
        ("p", is_p)
    )

    # --- funzioni ausiliarie
    P1v = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u_out_p1 = fem.Function(P1v)
    u_mesh_P1 = fem.Function(P1v)
    u_out = w.sub(0).collapse()

    plotter = plotter_q = None
    if flag_plot:
        plotter, update_animation = pltmd.create_elasticity_animation(
            W.sub(0), mesh, u_out, gdim, fem, filename="elastic_block.gif",
            show_edges=False, traslation=False, velocity=kin["v_p"], zoom=1.0)
    if flag_plot_q:
        plotter_q, update_q_animation = pltmd.create_qcriterion_animation(
            mesh, u_out, fem, filename="q_criterion.gif", q_threshold=10,
            traslation=True, above_color="yellow", below_color="green",
            show_mesh_outline=False, zoom=4.0)

    def step_solve(step):
        """Un passo: (1) aggiorna dati cinematici, (2) Laplace per u_mesh, (3) Navier-Stokes ALE."""
        # (1) aggiornamento dei dati al bordo e della forza apparente
        f.value = np.array([kin["a_p"][0]] + [0.0] * (gdim - 1))
        u_inlet.interpolate(inlet_field)
        u_wing_ns.interpolate(rotational_field_and_traslation_along_y)
        u_wing_md.interpolate(rotational_field_and_traslation_along_y)

        # (2) rotazione -> campo armonico di velocita' della mesh
        mesh_problem.solve()
        u_mesh.x.scatter_forward()

        # (3) Navier-Stokes
        ns_problem.solve()
        w.x.scatter_forward()

        reason = ns_problem.solver.getConvergedReason()
        n_its = ns_problem.solver.getIterationNumber()
        p_arr = w.sub(1).collapse().x.array
        if mesh.comm.rank == 0:
            print(f"step {step}, t = {t:.4f}, Newton its = {n_its}, reason = {reason}")
            print(f"max|w| = {np.max(np.abs(w.x.array)):.4e}, "
                  f"NaN: {np.isnan(w.x.array).any()}")
            print(f"p_mean={p_arr.mean():.6e}, p_min={p_arr.min():.6e}, "
                  f"p_max={p_arr.max():.6e}, range={p_arr.max() - p_arr.min():.6e}")
        return reason > 0

    # --- ciclo temporale
    F_t, F_t_, M_t, v_t, T_t, alpha_t, beta_t = [], [], [], [], [], [], []
    X = []
    saved_states = []
    alpha_pos = 0.0
    beta_pos = 0.0
    step = 0

    t += dt_val
    while t < T:
        step += 1
        kin["v_p"] = f1_t(t)
        kin["omega"] = f2_t(t)
        kin["a_p"] = f3_t(t)
        print(f"step {step}, t = {t:.4f}, v_x = {kin['v_p'][0]:.4f}, v_y = {kin['v_p'][1]:.4f}"
              f"omega = {kin['omega']:.4f}")

        if not step_solve(step):
            print(f"!! SNES non converso al passo {step}: interrompo e salvo i risultati parziali")
            break

        u_sol = w.sub(0).collapse()
        p_sol = w.sub(1).collapse()
        u_out_p1.interpolate(u_sol)
        X.append(u_out_p1.x.array.copy())

        # salvataggio stati a tempi prefissati
        tol = dt_val / 2
        for ts in save_times:
            if ts not in saved_states and abs(t - ts) < tol:
                np.savez(f"state_t{ts:.0f}.npz",
                         u_coords=V0.tabulate_dof_coordinates(), u=u_sol.x.array.copy(),
                         p_coords=Q0.tabulate_dof_coordinates(), p=p_sol.x.array.copy())
                saved_states.append(ts)

        # forze e momenti (calcolati PRIMA di muovere la mesh, sulla configurazione del passo)
        _, F_p, F_mu = compute_force_on_wing(u_sol, p_sol, mesh, TAG_WING, gdim, Re_inv, F_t, n, ds)
        F_t_.append((F_p, F_mu))

        # movimento mesh
        u_mesh_P1.interpolate(u_mesh)
        disp = u_mesh_P1.x.array.reshape((-1, gdim)) * dt_val
        mesh.geometry.x[:, :gdim] += disp

        # animazioni
        try:
            if flag_plot:
                update_animation(u_sol, disp, kin["v_p"], dt_val)
            if flag_plot_q:
                update_q_animation(u_sol, disp, kin["v_p"], dt_val)
        except Exception as e:
            print(f"Errore animazione al passo {step}: {e}")

        w_n.x.array[:] = w.x.array

        v_t.append(kin["v_p"])
        T_t.append(t)
        # alpha_pos += kin["v_p"] * dt_val
        # beta_pos += kin["omega"] * dt_val
        alpha_t.append(alpha_pos)
        beta_t.append(beta_pos)

        dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
        dt.value = dt_val
        t += dt_val

    # --- chiusura e post-processing
    if plotter is not None:
        plotter.close()
    if plotter_q is not None:
        plotter_q.close()

    if test:
        print(f"{np.array(T_t)}, {np.array(F_t)}, {np.array(v_t)}")
        # >>> qui rimetti IL TUO blocco con T_test, F_test, v_test (invariato) <<<
        T_test, F_test, v_test = ..., ..., ...
        pltmd.compare_forces(T_test, T_t, F_test, np.array(F_t), v_test, v_t)
        return

    T_t, v_t = np.array(T_t), np.array(v_t)
    F_t, M_t = np.array(F_t), np.array(M_t)
    alpha_t, beta_t = np.array(alpha_t), np.array(beta_t)
    F_parts = np.array([np.concatenate(pair) for pair in F_t_])   # [Fp(3), Fmu(3)]

    pltmd.plot_force(T_t, F_t, v_t)

    data = np.column_stack((T_t, v_t, F_t, M_t, alpha_t, beta_t, F_parts))
    header = ("time v_p force(3) torque alpha beta Fp(3) Fmu(3)\n"
              "config:\n" + json.dumps(cfg))
    np.savetxt("aerodynamic_history.txt", data, header=header, comments="# ")

    pltmd.save_POD(np.array(X), mesh.geometry.x, T_t)


if __name__ == "__main__":
    main()