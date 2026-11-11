import os
import sys
import json
import copy
from datetime import datetime

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# region Fenics_libraries
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
import ufl
from ufl import nabla_grad, Identity, grad, inner, dot, div
# endregion


# =====================================================================
# CONFIGURATION AND TRAJECTORY
# =====================================================================
def load_config(path):
    """Read JSON configuration and merge it section-by-section with defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as fh:
        user = json.load(fh)
    for section, values in user.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def make_output_dir(cfg, comm):
    """Create the run directory collectively and make it the working directory."""
    g, fl, tr, tm = cfg["geometry"], cfg["flow"], cfg["trajectory"], cfg["time"]
    name = None
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = (f"{stamp}_alpha{tr['alpha_deg']}_phi{tr['amp_phi_deg']}"
                f"_Re{fl['Re']}_res{g['resolution']}_dt{tm['dt_val_cost']}_L{g['L']}"
                "_p1p1_vms")
    name = comm.bcast(name, root=0)
    out_dir = os.path.abspath(os.path.join(cfg["output"]["base_dir"], name))
    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config_used.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)
    comm.Barrier()
    os.chdir(out_dir)
    return out_dir


def define_test_trajectory(omega, v_p, y0, alpha0, phi, decay):
    """Return ``custom_test`` kinematics without its plotting dependencies."""
    velocity = lambda t: np.array([
        v_p * (1.0 - np.exp(-decay * t)),
        -y0 * omega * np.sin(omega * t),
    ])
    angular_velocity = lambda t: (
        -alpha0
        * np.sin(omega * t + phi * (1.0 - np.exp(-decay * t)))
        * (omega + decay * phi * np.exp(-decay * t))
    )
    acceleration = lambda t: np.array([
        v_p * decay * np.exp(-decay * t),
        0.0,
    ])
    return velocity, angular_velocity, acceleration


# =====================================================================
# FINITE-ELEMENT FORMULATION
# =====================================================================
def NS_function_space(mesh, gdim):
    """Continuous equal-order ``[P1]^gdim x P1`` velocity-pressure space."""
    cell = mesh.basix_cell()
    P1_u = element("Lagrange", cell, 1, shape=(gdim,))
    P1_p = element("Lagrange", cell, 1)
    return fem.functionspace(mesh, mixed_element([P1_u, P1_p]))


def stabilization_parameters(mesh, c, nu, dt, C_I, C_t):
    """Compute VarMINT's metric-based ``tau_M`` and ``tau_C``.

    ``c = u - u_mesh`` is the ALE convective velocity.  If
    ``J = dx/dxi`` is the current cell Jacobian, VarMINT's convention gives
    ``G = 4 J^{-T} J^{-1}``.  The factor four comes from its use of
    ``0.5 * Jacobian(mesh)``.

    ``tau_M = (C_t/dt^2 + c.G.c + C_I*nu^2*G:G + eps)^(-1/2)``
    ``tau_C = 1 / (tau_M*tr(G))``

    The defaults ``C_I=3`` and ``C_t=4`` are intended for P1 elements.
    """
    J_inv = ufl.JacobianInverse(mesh)
    G = 4.0 * dot(J_inv.T, J_inv)
    denominator_squared = (
        C_t / dt**2
        + dot(c, dot(G, c))
        + C_I * nu**2 * inner(G, G)
        + np.finfo(PETSc.RealType).eps
    )
    tau_M = 1.0 / ufl.sqrt(denominator_squared)
    tau_C = 1.0 / (tau_M * ufl.tr(G))
    return tau_M, tau_C


def define_form_NS(
    Re_inv, u, p, v, q, f, u_n, u_mesh, dt, mesh, C_I, C_t
):
    """Backward-Euler ALE Navier--Stokes residual with full VarMINT VMS.

    The modeled fine scales are ``u'=-tau_M*r_M`` and ``p'=-tau_C*r_C``,
    where ``r_M`` and ``r_C`` are the strong momentum and continuity
    residuals.  The four labeled additions below are the only stabilization
    terms introduced by this port.
    """
    c = u - u_mesh
    twice_strain = nabla_grad(u) + nabla_grad(u).T
    sigma = -p * Identity(u.ufl_shape[0]) + Re_inv * twice_strain
    material_acceleration = (u - u_n) / dt + dot(c, nabla_grad(u))

    # Original Galerkin formulation, written in equivalent stress form.
    F = (
        inner(material_acceleration - f, v)
        + inner(sigma, grad(v))
        + q * div(u)
    ) * ufl.dx

    # Strong element residuals used to model the unresolved scales.
    r_M = material_acceleration - div(sigma) - f
    r_C = div(u)
    tau_M, tau_C = stabilization_parameters(
        mesh, c, Re_inv, dt, C_I, C_t
    )
    u_prime = -tau_M * r_M
    p_prime = -tau_C * r_C

    F += (
        # SUPG + PSPG: convection and equal-order pressure stabilization.
        -inner(dot(c, nabla_grad(v)) + grad(q), u_prime)
        # LSIC / grad-div stabilization.
        - p_prime * div(v)
        # VMS convective cross stress.
        + inner(v, dot(u_prime, nabla_grad(u)))
        # VMS Reynolds stress.
        - inner(grad(v), ufl.outer(u_prime, u_prime))
    ) * ufl.dx
    return F


def enforce_zero_mean_pressure(
    w, pressure_to_mixed, pressure_integral_form, volume_form, mesh
):
    """Select the pressure gauge ``integral_Omega(p) dx = 0``.

    PSPG stabilizes non-constant P1/P1 pressure modes but does not remove the
    physical nullspace ``p -> p + constant``.  MUMPS first solves that singular
    system using ICNTL(24)=1 and ICNTL(25)=0.  This function then subtracts the
    MPI-global domain average before diagnostics, forces, output, and the next
    time step.  Velocity and pressure gradients are unchanged.
    """
    local_volume = fem.assemble_scalar(volume_form)
    local_pressure_integral = fem.assemble_scalar(pressure_integral_form)
    volume = mesh.comm.allreduce(local_volume, op=MPI.SUM)
    pressure_integral = mesh.comm.allreduce(
        local_pressure_integral, op=MPI.SUM
    )
    pressure_mean = pressure_integral / volume
    w.x.array[pressure_to_mixed] -= pressure_mean
    w.x.scatter_forward()
    return pressure_mean


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


def eval_dt(f1_t, f2_t, dt_val_cost, t):
    v_max = max(np.max(np.abs(f1_t(t))), np.max(np.abs(f2_t(t))), 1)
    return dt_val_cost / (1 + v_max)


def compute_force_on_wing(u_h, p_h, mesh, mt, wing_tag, gdim, Re_inv, F_t):
    """Evaluate pressure and viscous force on the two-dimensional wing."""
    I = Identity(gdim)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    j = ufl.indices(1)[0]
    Fx_p = -np.array([
        fem.assemble_scalar(fem.form(-p_h * I[i, j] * n[j] * ds(wing_tag)))
        for i in range(gdim)
    ])
    Fx_mu = -np.array([
        fem.assemble_scalar(fem.form(
            Re_inv
            * (nabla_grad(u_h) + nabla_grad(u_h).T)[i, j]
            * n[j]
            * ds(wing_tag)
        ))
        for i in range(gdim)
    ])
    Fx_ = Fx_p + Fx_mu
    if mesh.comm.rank == 0:
        print("  forza sul corpo interno:", Fx_,
              " (pressione:", Fx_p, ", viscosita':", Fx_mu, ")")
        F_t.append(Fx_)
    return Fx_, Fx_p, Fx_mu


# =====================================================================
# CONFIGURATION
# =====================================================================
DEFAULTS = {
    "geometry": {
        "L": 6.0,
        "r_x": 0.5,
        "r_y": 0.02,
        "N2": 100,
        "resolution": 0.01,
    },
    "flow": {"Re": 100},
    "time": {"T": 0.75, "t_start": 0.0, "dt_val_cost": 0.07},
    "trajectory": {
        "alpha_deg": 40,
        "amp_phi_deg": 100,
        "period_phi": 0.5,
        "period_alpha": 0.5,
        "phase_alpha": 0.72,
        "sigma_phi": None,
        "sigma_alpha": None,
        "n_grid": 4096,
        "num_steps": 80,
    },
    # VarMINT defaults for continuous linear elements.
    "stabilization": {"C_I": 3.0, "C_t": 4.0},
    "output": {"base_dir": "runs", "save_times": [1.0, 2.0, 3.0, 4.0]},
}


# =====================================================================
# MAIN
# =====================================================================
def main():
    config_path = os.path.abspath(
        sys.argv[1] if len(sys.argv) > 1 else "config_test.json"
    )
    cfg = load_config(config_path)

    # --- parameters
    g, tm, tr = cfg["geometry"], cfg["time"], cfg["trajectory"]
    st = cfg["stabilization"]
    L, r_x, r_y, N2, resolution = (
        g["L"], g["r_x"], g["r_y"], g["N2"], g["resolution"]
    )
    Re = cfg["flow"]["Re"]
    T, t, dt_val_cost = tm["T"], tm["t_start"], tm["dt_val_cost"]
    alpha_deg, amp_phi_deg = tr["alpha_deg"], tr["amp_phi_deg"]
    alpha_amp = np.deg2rad(alpha_deg)
    amp_phi = np.deg2rad(amp_phi_deg)
    mesh_angle = np.pi + alpha_amp
    N = int(4 * r_x / resolution)
    save_times = cfg["output"]["save_times"]
    c_r = np.array(g["c_r"], dtype=float)
    U = 1

    frequency = tr["St"] / (2 * 0.75)
    omega_angular = 2 * np.pi * frequency
    T = T / frequency
    dt_val_cost = dt_val_cost / frequency

    # --- output directory (cwd changes here)
    out_dir = make_output_dir(cfg, MPI.COMM_WORLD)
    print("Output in:", out_dir)

    # Import locally because mesh generation has optional visualization
    # dependencies that are unrelated to the VMS form itself.
    from mesh_generation import generate_mesh_circular_naca

    generate_mesh_circular_naca(
        R=L,
        N2=N2,
        chord=1,
        alpha=mesh_angle,
        c_r=c_r,
        t=0.12,
        n_pts=g["N2"],
        sizes=g["sizes"],
        dists=g["dists"],
        mesh_file="wing.msh",
    )

    f1_t, f2_t, f3_t = define_test_trajectory(
        omega_angular, U, tr["y0"], alpha_amp, amp_phi, tr["lamb"]
    )

    # region Import mesh
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    gdim = mesh.geometry.dim
    with XDMFFile(mesh.comm, "mf.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, name="Grid")
    # endregion

    # region Function spaces
    W = NS_function_space(mesh, gdim)
    w = fem.Function(W)
    w_n = fem.Function(W)
    u, p = ufl.split(w)
    u_n, _ = ufl.split(w_n)
    v, q = ufl.TestFunctions(W)
    V0, _ = W.sub(0).collapse()
    Q0, pressure_to_mixed = W.sub(1).collapse()
    V0_x, _ = W.sub(0).sub(0).collapse()
    V0_y, _ = W.sub(0).sub(1).collapse()
    # endregion

    dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
    Re_inv = fem.Constant(mesh, PETSc.ScalarType(1 / Re))
    dt = fem.Constant(mesh, PETSc.ScalarType(dt_val))

    # Compile the pressure-gauge integrals once.  They are reassembled on the
    # current mesh after every nonlinear solve.
    one = fem.Constant(mesh, PETSc.ScalarType(1.0))
    volume_form = fem.form(one * ufl.dx)
    pressure_integral_form = fem.form(p * ufl.dx)

    # region Initial boundary conditions
    dofs = dofs_from_tags_NS([2, 3], W, mt, fdim)
    bcs0 = [
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0][0][0], W.sub(0).sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0][1][0], W.sub(0).sub(1)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[1][0][0], W.sub(0).sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs[1][1][0], W.sub(0).sub(1)),
    ]
    # endregion

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
        # PSPG leaves the constant-pressure nullspace unchanged.  These are
        # the DOLFINx-recommended MUMPS settings for the singular Jacobian.
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
        "ksp_error_if_not_converged": True,
    }

    # region Auxiliary functions for the time loop
    P1v = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u_out_p1 = fem.Function(P1v)
    V_geom = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u_mesh_P1 = fem.Function(V_geom)

    u_mesh = fem.Function(V0)
    u_flow = fem.Function(V0_x)
    u_wing_x = fem.Function(V0_x)
    u_wing_y = fem.Function(V0_y)
    # endregion

    kin = {
        "omega": 0.0,
        "v_p": np.array([0.0, 0.0]),
        "a_p": np.array([0.0, 0.0]),
        "c_p": np.array([0.0, 0.0]),
    }

    # region Fields used by boundary conditions
    def wing_field_x(xx):
        return -kin["omega"] * (xx[1] - kin["c_p"][1]) + kin["v_p"][0]

    def wing_field_y(xx):
        return kin["omega"] * (xx[0] - kin["c_p"][0]) + kin["v_p"][1]

    def rotational_field(xx):
        values = np.zeros((gdim, xx.shape[1]), dtype=PETSc.ScalarType)
        values[0] = (
            -kin["omega"] * (xx[1] - kin["c_p"][1]) + kin["v_p"][0]
        )
        values[1] = (
            kin["omega"] * (xx[0] - kin["c_p"][0]) + kin["v_p"][1]
        )
        return values
    # endregion

    def step_solve(step):
        """Solve one time step and return whether PETSc SNES converged."""
        u_mesh.interpolate(rotational_field)
        u_mesh.x.scatter_forward()
        body_force = fem.Constant(mesh, PETSc.ScalarType((0.0,) * gdim))

        # region Update boundary conditions
        u_flow.x.array[:] = 0
        u_wing_x.interpolate(wing_field_x)
        u_wing_y.interpolate(wing_field_y)
        bcs = [
            fem.dirichletbc(u_flow, dofs[0][0], W.sub(0).sub(0)),
            fem.dirichletbc(
                PETSc.ScalarType(0.0), dofs[0][1][0], W.sub(0).sub(1)
            ),
            fem.dirichletbc(u_wing_x, dofs[1][0], W.sub(0).sub(0)),
            fem.dirichletbc(u_wing_y, dofs[1][1], W.sub(0).sub(1)),
        ]
        # endregion

        F = define_form_NS(
            Re_inv, u, p, v, q, body_force, u_n, u_mesh, dt, mesh,
            st["C_I"], st["C_t"]
        )
        problem = NonlinearProblem(
            F,
            w,
            bcs=bcs,
            petsc_options_prefix="navier_stokes_vms_",
            petsc_options=petsc_options,
        )
        problem.solve()
        w.x.scatter_forward()

        removed_pressure_mean = enforce_zero_mean_pressure(
            w,
            pressure_to_mixed,
            pressure_integral_form,
            volume_form,
            mesh,
        )

        # region Diagnostic output
        reason = problem.solver.getConvergedReason()
        n_its = problem.solver.getIterationNumber()
        p_arr = w.sub(1).collapse().x.array
        if mesh.comm.rank == 0:
            print(
                f"step {step}, t = {t:.4f}, Newton its = {n_its}, "
                f"reason = {reason}, removed p_mean = "
                f"{removed_pressure_mean:.6e}"
            )
            print(
                f"max|w| = {np.max(np.abs(w.x.array)):.4e}, "
                f"NaN: {np.isnan(w.x.array).any()}"
            )
            print(
                f"p_coefficient_mean={p_arr.mean():.6e}, p_min={p_arr.min():.6e}, "
                f"p_max={p_arr.max():.6e}, "
                f"range={p_arr.max() - p_arr.min():.6e}"
            )
        # endregion

        return reason > 0

    # region Time-loop initialization
    F_t, F_t_, v_t, T_t, alpha_t, beta_t = [], [], [], [], [], []
    alpha_pos = np.zeros(2)
    beta_pos = 0.0
    step = 1
    # endregion

    t += dt_val
    while t < T:
        kin["v_p"] = f1_t(t)
        kin["omega"] = f2_t(t)
        kin["a_p"] = f3_t(t)
        print(
            f"step {step}, t = {t:.4f}, v_px = {kin['v_p'][0]:.4f}, "
            f"v_py = {kin['v_p'][1]:.4f}, omega = {kin['omega']:.4f}"
        )

        if not step_solve(step):
            print(
                f"!! SNES non converged at step {step}: "
                "stopping and saving partial results"
            )
            break

        # region Save state
        u_sol = w.sub(0).collapse()
        p_sol = w.sub(1).collapse()
        u_out_p1.interpolate(u_sol)

        p_sol.name = "pressure"
        filename = f"solution_{step:06d}.xdmf"
        with XDMFFile(mesh.comm, filename, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u_out_p1, t)
            xdmf.write_function(p_sol, t)
        # endregion

        # Move mesh exactly as in the original solver.
        u_mesh_P1.interpolate(u_mesh)
        mesh.geometry.x[:, :gdim] += (
            u_mesh_P1.x.array.reshape((-1, gdim)) * dt_val
        )
        kin["c_p"] += kin["v_p"] * dt_val
        alpha_pos += kin["v_p"] * dt_val
        beta_pos += kin["omega"] * dt_val

        w_n.x.array[:] = w.x.array

        # Forces
        _, F_p, F_mu = compute_force_on_wing(
            u_sol, p_sol, mesh, mt, 3, gdim, Re_inv, F_t
        )

        # region Save variables
        F_t_.append((F_p, F_mu))
        v_t.append(kin["v_p"])
        T_t.append(t)
        alpha_t.append(alpha_pos.copy())
        beta_t.append(beta_pos)
        # endregion

        dt_val = eval_dt(f1_t, f2_t, dt_val_cost, t)
        dt.value = dt_val
        t += dt_val
        step += 1

    # Save numerical history only on rank zero.  Force components are flattened
    # to keep the text output two-dimensional.
    if mesh.comm.rank == 0 and T_t:
        force_parts = np.array([
            np.concatenate((pressure_part, viscous_part))
            for pressure_part, viscous_part in F_t_
        ])
        data = np.column_stack((
            np.asarray(T_t),
            np.asarray(v_t),
            np.asarray(F_t),
            np.asarray(alpha_t),
            np.asarray(beta_t),
            force_parts,
        ))
        header = (
            "time v_p force alpha beta pressure_force viscous_force\n"
            "config:\n" + json.dumps(cfg)
        )
        np.savetxt(
            "aerodynamic_history.txt", data, header=header, comments="# "
        )


if __name__ == "__main__":
    main()
