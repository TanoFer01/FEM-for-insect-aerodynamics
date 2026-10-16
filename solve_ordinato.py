import os
import sys
import json
import copy
import shutil
from datetime import datetime

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.integrate import cumulative_trapezoid, test
import matplotlib.pyplot as plt

from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
import ufl
from ufl import nabla_grad, Identity, grad, inner, dot, div

sys.path.append("/home/ferra/wing_dynamics/modules")
import plot as pltmd
from mesh_generation import generate_mesh_circular
from estrazione_spline import crea_spline_estesa, crea_funzioni
from custom_fun import custom_sin, zanzara, custom_di


# =====================================================================
# CONFIGURAZIONE, preferibilmente cambiare i parametri nel file config.json invece che qui
# =====================================================================
DEFAULTS = {
    "geometry": {"L": 6.0, "r_x": 0.5, "r_y": 0.02, "N2": 100, "resolution": 0.01}, #Parametri geometrici generazione mesh: L raggio esterno dominio circolare, r_x metà lunghezza ala, r_y spessore ala, N2 numero di punti lungo il bordo esterno del dominio circolare, resolution risoluzione della mesh al bordo dell'ala, più ci si allontana più la risoluzione aumenta gradualmente  
    "flow": {"Re": 100}, #Numero di Reynolds
    "time": {"T": 0.75, "t_start": 0.0, "dt_val_cost": 0.07}, #T tempo finale di simulazione, t_start tempo iniziale di simulazione, dt_val_cost passo temporale massimo
    "trajectory": {  #parametri traiettoria, generati da un file esterno
        "alpha_deg": 40, "amp_phi_deg": 100, 
        "period_phi": 0.5, "period_alpha": 0.5, "phase_alpha": 0.72,
        "sigma_phi": None, "sigma_alpha": None,
        "n_grid": 4096, "num_steps": 80, 
    },
    "flags": {"flag_plot": True, "flag_plot_q": True, "start_from_previous": False}, #flag per generazione animazioni e per riprendere la simulazione da uno stato precedente (quest'ultimo attualmente non è implementato)
    "output": {"base_dir": "runs", "save_times": [1.0, 2.0, 3.0, 4.0]}, #base_dir cartella di output, save_times lista dei tempi in cui salvare lo stato della simulazione
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
    """
    Crea la cartella  <base_dir>/<data_ora>_<parametri principali>  e ci entra (chdir):
    da quel momento mesh, gif, stati salvati, plot e risultati finiscono tutti li'.
    """
    g, fl, tr, tm = cfg["geometry"], cfg["flow"], cfg["trajectory"], cfg["time"]
    name = None
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = (f"{stamp}_alpha{tr['alpha_deg']}_phi{tr['amp_phi_deg']}"
                f"_Re{fl['Re']}_res{g['resolution']}_dt{tm['dt_val_cost']}_L{g['L']}")
    name = comm.bcast(name, root=0)          # stesso nome su tutti i rank
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
        inner((u - u_n) / dt, v) * ufl.dx                              # transitorio
        + inner(dot(u - u_mesh, nabla_grad(u)), v) * ufl.dx            # convezione (ALE)
        + Re_inv * inner(nabla_grad(u) + nabla_grad(u).T, grad(v)) * ufl.dx  # viscosita'
        - div(v) * p * ufl.dx                                          # pressione
        + q * div(u) * ufl.dx                                          # incomprimibilita'
        - inner(f, v) * ufl.dx                                         # forzante
    )


def eval_dt(f1_t, f2_t, dt_val_cost, t):
    v_max = max(np.max(np.abs(f1_t(t))), np.max(np.abs(f2_t(t))), 1)
    return dt_val_cost / (1 + v_max)


def compute_force_on_wing(u_h, p_h, mesh, mt, wing_tag, gdim, Re_inv, F_t): #valutazione forza su profilo 2D
    I = Identity(gdim)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    j = ufl.indices(1)[0]
    sigma = -p_h * I + Re_inv * (nabla_grad(u_h) + nabla_grad(u_h).T)
    Fx_p = -np.array([
        fem.assemble_scalar(fem.form(-p_h * I[i,j] * n[j] * ds(wing_tag)))
        for i in range(gdim)
    ])
    Fx_mu = -np.array([
            fem.assemble_scalar(fem.form(Re_inv * (nabla_grad(u_h) + nabla_grad(u_h).T)[i, j] * n[j] * ds(wing_tag)))
            for i in range(gdim)
        ])
    Fx_ = Fx_p + Fx_mu
    if mesh.comm.rank == 0:
        print("  forza sul corpo interno:", Fx_, " (pressione:", Fx_p, ", viscosita':", Fx_mu, ")")
        F_t.append(Fx_)
    return Fx_, Fx_p, Fx_mu


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
# MAIN
# =====================================================================
def main():
    #Set true to run the test case, or False to run the normal simulation
    test = True
    if test == True:
        config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "config_test.json")
    else:
        config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "config.json")

    cfg = load_config(config_path)
    

    # --- parametri
    g, tm, tr, fl = cfg["geometry"], cfg["time"], cfg["trajectory"], cfg["flags"]
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
    save_times = cfg["output"]["save_times"]

    # --- cartella di output (da qui in poi cwd = cartella del run)
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
    ) #estrazione della traiettoria da file esterno, f1_t = dphi/dt, f2_t = dalpha/dt, f3_t = d2phi/dt2

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    gdim = mesh.geometry.dim
    with XDMFFile(mesh.comm, "mf.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, name="Grid")

    # --- spazi e funzioni
    W = NS_function_space(mesh, gdim)
    w = fem.Function(W)          # n+1
    w_n = fem.Function(W)        # n
    (u, p) = ufl.split(w)
    (u_n, _) = ufl.split(w_n)
    (v, q) = ufl.TestFunctions(W)

    dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
    Re_inv = fem.Constant(mesh, PETSc.ScalarType(1 / Re))
    dt = fem.Constant(mesh, PETSc.ScalarType(dt_val))

    # --- BC iniziali (usate solo per consistenza dello stato iniziale)
    dofs = dofs_from_tags_NS([2, 3], W, mt, fdim)   # 2 = bordo esterno, 3 = ala
    bcs0 = [
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0][0][0], W.sub(0).sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0][1][0], W.sub(0).sub(1)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[1][0][0], W.sub(0).sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[1][1][0], W.sub(0).sub(1)),
    ]

    if fl["start_from_previous"]:
        data = np.load("state_t1.npz")
        mesh.geometry.x[:] = data["coords"]
        u_prev, p_prev = w_n.split()
        u_prev.x.array[:] = data["u"]
        p_prev.x.array[:] = data["p"]
        u_prev.x.scatter_forward()
        p_prev.x.scatter_forward()
        w.x.array[:] = w_n.x.array
        w.x.scatter_forward()
    else:
        w.x.array[:] = 0.0
        w_n.x.array[:] = 0.0
        fem.set_bc(w.x.array, bcs0)
        fem.set_bc(w_n.x.array, bcs0)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-7,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_linesearch_type": "bt",
        "snes_linesearch_damping": 0.8,
    }


    # --- funzioni ausiliarie per ciclo temporale
    P1v = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u_out_p1 = fem.Function(P1v)
    V_geom = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u_mesh_P1 = fem.Function(V_geom)
    V0, _ = W.sub(0).collapse()
    Q0, _ = W.sub(1).collapse()
    u_mesh = fem.Function(V0)
    V0_x, _ = W.sub(0).sub(0).collapse()
    V0_y, _ = W.sub(0).sub(1).collapse()
    u_flow = fem.Function(V0_x)
    u_wing_x = fem.Function(V0_x)
    u_wing_y = fem.Function(V0_y)

    u_out = w.sub(0).collapse()
    plotter = plotter_q = None
    if flag_plot:
        plotter, update_animation = pltmd.create_elasticity_animation(
            W.sub(0), mesh, u_out, gdim, fem, filename="elastic_block.gif",
            show_edges=False, traslation=True, velocity=0)
    if flag_plot_q:
        plotter_q, update_q_animation = pltmd.create_qcriterion_animation(
            mesh, u_out, fem, filename="q_criterion.gif", q_threshold=10,
            traslation=True, above_color="yellow", below_color="green",
            show_mesh_outline=False, zoom=4.0)

    # --- cinematica corrente (letta dai campi di velocita' di corpo rigido)
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

    def step_solve(step):
        """Un passo temporale. Ritorna True se Newton e' convergito."""
        u_mesh.interpolate(rotational_field)
        f = fem.Constant(mesh, PETSc.ScalarType((kin["a_p"],) + (0.0,) * (gdim - 1)))

        u_flow.x.array[:] = -kin["v_p"]
        u_wing_x.interpolate(rotational_field_x)
        u_wing_y.interpolate(rotational_field_y)
        bcs = [
            fem.dirichletbc(u_flow, dofs[0][0], W.sub(0).sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0][1][0], W.sub(0).sub(1)),
            fem.dirichletbc(u_wing_x, dofs[1][0], W.sub(0).sub(0)),
            fem.dirichletbc(u_wing_y, dofs[1][1], W.sub(0).sub(1)),
        ]

        F = define_form_NS(Re_inv, u, p, v, q, f, u_n, u_mesh, dt)
        problem = NonlinearProblem(F, w, bcs=bcs,
                                   petsc_options_prefix="navier_stokes_",
                                   petsc_options=petsc_options)
        problem.solve()
        u_mesh.x.scatter_forward()

        reason = problem.solver.getConvergedReason()
        n_its = problem.solver.getIterationNumber()
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
    alpha_pos = 0.0     # posizione traslazionale integrata
    beta_pos = 0.0      # angolo integrato
    step = 0

    t += dt_val
    while t < T:
        step += 1
        kin["v_p"] = -f1_t(t)[0]
        kin["omega"] = f2_t(t)[0]
        kin["a_p"] = -f3_t(t)[0]
        print(f"step {step}, t = {t:.4f}, v_p = {kin['v_p']:.4f}, "
              f"omega = {kin['omega']:.4f}, a_p = {kin['a_p']:.4f}")

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

        # movimento mesh
        u_mesh_P1.interpolate(u_mesh)
        mesh.geometry.x[:, :gdim] += u_mesh_P1.x.array.reshape((-1, gdim)) * dt_val

        # animazioni
        try:
            disp = u_mesh_P1.x.array.reshape((-1, gdim)) * dt_val
            if flag_plot:
                update_animation(u_sol, disp, kin["v_p"], dt_val)
            if flag_plot_q:
                update_q_animation(u_sol, disp, kin["v_p"], dt_val)
        except Exception as e:
            print(f"Errore animazione al passo {step}: {e}")

        w_n.x.array[:] = w.x.array

        # forze e momenti
        _, F_p, F_mu = compute_force_on_wing(u_sol, p_sol, mesh, mt, 3, gdim, Re_inv, F_t)
        compute_torque_on_wing(u_sol, p_sol, mesh, mt, 3, gdim, Re_inv, M_t, kin["c_p"])

        F_t_.append((F_p, F_mu))

        v_t.append(kin["v_p"])
        T_t.append(t)
        alpha_pos += kin["v_p"] * dt_val
        beta_pos += kin["omega"] * dt_val
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


    if test == True:
        print(f"{np.array(T_t)}, {np.array(F_t)}, {np.array(v_t)}")
        T_test, F_test, v_test = [0.01, 0.02, 0.03, 0.04, np.float64(0.049198043592108984), np.float64(0.054880460834539804), np.float64(0.058426115913673395), np.float64(0.061085263394691156), np.float64(0.06324866463088945), np.float64(0.06509223504367033), np.float64(0.06671116124086061), np.float64(0.06816311111100237), np.float64(0.06948582500023577), np.float64(0.07070543213548766), np.float64(0.071840817841171), np.float64(0.07290610382042306), np.float64(0.07391214523565445), np.float64(0.07486747952883104), np.float64(0.07577895233480966), np.float64(0.07665214444222598), np.float64(0.07749167144437803), np.float64(0.07830139925280623), np.float64(0.07908460243602422), np.float64(0.07984408275083597), np.float64(0.0805822593599091), np.float64(0.08130123852579761), np.float64(0.0820028681747239), np.float64(0.08268878113574828), np.float64(0.08336042978705624), np.float64(0.08401911410096184), np.float64(0.0846660045603488), np.float64(0.08530216104976437), np.float64(0.08592854855760905), np.float64(0.0865460503306031), np.float64(0.08715547897715287), np.float64(0.0877575859080449), np.float64(0.08835306942105607), np.float64(0.08894258167360646), np.float64(0.08952673473950838)], np.array([np.array([ 3.64974488,  3.54804289, -0.        ]), np.array([10.85817013, 10.88625885, -0.        ]), np.array([20.26779694, 20.75336028, -0.        ]), np.array([39.65045306, 41.8074422 , -0.        ]), np.array([88.60218998, 92.99880656, -0.        ]), np.array([167.21354311, 170.15604425,  -0.        ]), np.array([245.52293819, 240.79835058,  -0.        ]), np.array([316.31337381, 298.10676789,  -0.        ]), np.array([384.35031659, 348.02167825,  -0.        ]), np.array([450.4733132 , 391.87928942,  -0.        ]), np.array([515.0595547 , 430.44041612,  -0.        ]), np.array([578.27348855, 464.19546842,  -0.        ]), np.array([640.17322352, 493.49491474,  -0.        ]), np.array([700.76092628, 518.611306  ,  -0.        ]), np.array([760.00010389, 539.76499889,  -0.        ]), np.array([817.82561391, 557.14059993,  -0.        ]), np.array([874.15376428, 570.90081158,  -0.        ]), np.array([928.88411975, 581.19203408,  -0.        ]), np.array([981.93769367, 588.17153761,  -0.        ]), np.array([1033.25220636,  592.00148015,   -0.        ]), np.array([1082.72531795,  592.81392722,   -0.        ]), np.array([1130.23811703,  590.72954412,   -0.        ]), np.array([1175.68187716,  585.87461575,   -0.        ]), np.array([1218.94080832,  578.37257144,   -0.        ]), np.array([1259.91651172,  568.35820441,   -0.        ]), np.array([1298.52838188,  555.97651067,   -0.        ]), np.array([1334.67786964,  541.36452746,   -0.        ]), np.array([1368.26273865,  524.65809944,   -0.        ]), np.array([1399.20579679,  506.00300775,   -0.        ]), np.array([1427.45751814,  485.55379944,   -0.        ]), np.array([1452.98674024,  463.46848865,   -0.        ]), np.array([1475.75527773,  439.89900756,   -0.        ]), np.array([1495.7043689 ,  414.98880322,   -0.        ]), np.array([1512.77339187,  388.8817416 ,   -0.        ]), np.array([1526.90990885,  361.72603469,   -0.        ]), np.array([1538.06519043,  333.67276583,   -0.        ]), np.array([1546.21226372,  304.87962113,   -0.        ]), np.array([1551.35909103,  275.51091007,   -0.        ]), np.array([1553.51262321,  245.72642242,   -0.        ])]), [np.float64(-0.08308091989643931), np.float64(-0.32413896301973755), np.float64(-0.6997413916545214), np.float64(-1.1743754310055707), np.float64(-1.6615517661510246), np.float64(-1.9709094814530514), np.float64(-2.1627512659040833), np.float64(-2.3046661619019715), np.float64(-2.418311538599435), np.float64(-2.5135668322167506), np.float64(-2.5958228246508868), np.float64(-2.6683641532630666), np.float64(-2.733348945454689), np.float64(-2.7922754123692632), np.float64(-2.8462283014831624), np.float64(-2.8960195198202143), np.float64(-2.9422733371913337), np.float64(-2.9854805561579805), np.float64(-3.0260343393939464), np.float64(-3.0642547009774344), np.float64(-3.1004057242388328), np.float64(-3.1347079611812565), np.float64(-3.1673475503964097), np.float64(-3.1984830455186435), np.float64(-3.2282506118966214), np.float64(-3.256768037885013), np.float64(-3.284137870138043), np.float64(-3.310449891346687), np.float64(-3.335783097237565), np.float64(-3.360207287090158), np.float64(-3.383784352111439), np.float64(-3.4065693246803477), np.float64(-3.428611235981579), np.float64(-3.4499538181812124), np.float64(-3.4706360788352546), np.float64(-3.490692768818532), np.float64(-3.5101547601660426), np.float64(-3.529049346419115), np.float64(-3.547400475074468)]
        pltmd.compare_forces(T_test,T_t, F_test,np.array(F_t), v_test, v_t)
        return
    
    T_t, v_t = np.array(T_t), np.array(v_t)
    F_t, M_t = np.array(F_t), np.array(M_t)
    alpha_t, beta_t = np.array(alpha_t), np.array(beta_t)

    pltmd.plot_force(T_t, F_t, v_t)

    data = np.column_stack((T_t, v_t, F_t, M_t, alpha_t, beta_t, F_t_))
    header = ("time v_p force torque alpha beta\n"
              "config:\n" + json.dumps(cfg))
    np.savetxt("aerodynamic_history.txt", data, header=header, comments="# ")

    pltmd.save_POD(np.array(X), mesh.geometry.x, T_t)


if __name__ == "__main__":
    main()