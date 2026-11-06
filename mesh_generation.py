import pygmsh
import gmsh
import meshio
import numpy as np
import pyvista as pv
from dolfinx import plot




def generate_mesh_circular(R, r_x, r_y, N, N2, alpha, c_r, resolution):

    def rotate(x, y, alpha):
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])
        r = R @ np.array([x, y])
        return (r[0], r[1], 0)

    mesh_file = "wing.msh"
    mesh_size = 2*np.pi*R/N2

    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    circle_out = [
                model.add_point(
                    rotate(R * np.cos(i * 2 * np.pi / N2),
                        R * np.sin(i * 2 * np.pi / N2), alpha),
                    mesh_size=mesh_size,
                )
                for i in range(N2)
            ]
    circle_lines_out = [
            model.add_line(circle_out[i], circle_out[(i + 1) % N2]) for i in range(N2)
        ]

    dx = resolution#np.sqrt(resolution**2 - r_y**2)
    points_wing = [(r_x*(1+dx)+ c_r[0], c_r[1]), (r_x+ c_r[0], r_y+ c_r[1]), (-r_x+ c_r[0], r_y+ c_r[1]),(-r_x*(1+dx)+ c_r[0], c_r[1]), (-r_x+ c_r[0], -r_y+ c_r[1]), (r_x+ c_r[0], -r_y+ c_r[1])] #[(i/N*r_x - r_x + c_r[0], r_y/2 + c_r[1]) for i in range(N)] + [(r_x*1/N + c_r[0], 0 + c_r[1])] +[(-i/N*r_x + r_x + c_r[0], -r_y/2 + c_r[1]) for i in range(N)] + [(-r_x*1/N + c_r[0], 0 + c_r[1])]
    points_wing = [(r_x + c_r[0], c_r[1]), ( c_r[0], r_y+ c_r[1]), (-r_x+ c_r[0], c_r[1]),(c_r[0], c_r[1] - r_y)]

    # circle_r = [
    #         model.add_point(
    #             rotate(c_r[0] + r_x * np.cos(i * 2 * np.pi / N),
    #                 c_r[1] + r_y * np.sin(i * 2 * np.pi / N), alpha),
    #             mesh_size=mesh_size,
    #         )
    #         for i in range(N)
    #     ]

    circle_r = [
                model.add_point(
                    rotate(points_wing[i][0],
                        points_wing[i][1], alpha),
                    mesh_size=resolution,
                )
                for i in range(len(points_wing))
            ]

    circle_lines = [
        model.add_line(circle_r[i], circle_r[(i + 1) % len(circle_r)]) for i in range(len(circle_r))
    ]

    channel_loop = model.add_curve_loop(circle_lines_out)
    circle_loop = model.add_curve_loop(circle_lines)

    plane_surface = model.add_plane_surface(channel_loop, holes=[circle_loop])
    gmsh.model.geo.synchronize()

    # # ----------------------------------------------------------------------
    # # Mesh size parameters
    # # ----------------------------------------------------------------------

    # # Mesh sul profilo
    # eps_wall = resolution

    # # Mesh massima nel dominio
    # eps_out = mesh_size

    # # Distanza entro cui la mesh cresce dal profilo
    # R_wall = R-2*r_x

    # # ----------------------------------------------------------
    # # Raffinamenti locali
    # #
    # # center    : centro del raffinamento
    # # radius    : raggio della zona più raffinata
    # # thickness : spessore della transizione
    # # size      : dimensione della mesh all'interno
    # # ----------------------------------------------------------

    # local_refinements = [
    #     {
    #                 "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #                 "radius": 5*r_x,
    #                 "thickness": 0.25,
    #                 "size": eps_wall,
    #             },
        
    #             {
    #                         "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #                         "radius": 7*r_x,
    #                         "thickness": 0.25,
    #                         "size": eps_wall*2,
    #                     }

    #     # {
    #     #     "center": (rotate(r_x+c_r[0], c_r[1], alpha)[0], rotate(r_x+c_r[0], c_r[1], alpha)[1]),
    #     #     "radius": 3*r_x,
    #     #     "thickness": 0.25,
    #     #     "size": eps_wall,
    #     # },

    #     # {
    #     #             "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #     #             "radius": 6*r_x,
    #     #             "thickness": 0.25,
    #     #             "size": eps_wall*10,
    #     #         },

    #     # {
    #     #             "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #     #             "radius": 1*r_x,
    #     #             "thickness": 0.25,
    #     #             "size": eps_wall/2,
    #     #         },

    #     # {
    #     #             "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #     #             "radius": 3*r_x,
    #     #             "thickness": 0.25,
    #     #             "size": eps_wall,
    #     #         },
    #     # {
    #     #                             "center": (rotate(r_x+c_r[0], c_r[1], alpha)[0], rotate(r_x+c_r[0], c_r[1], alpha)[1]),
    #     #                             "radius": 1*r_x,
    #     #                             "thickness": 0.25,
    #     #                             "size": eps_wall/2,
    #     #                         },
    #     #                         {
    #     #                                             "center": (rotate(-r_x+c_r[0], c_r[1], alpha)[0], rotate(-r_x+c_r[0], c_r[1], alpha)[1]),
    #     #                                             "radius": 0.5*r_x,
    #     #                                             "thickness": 0.25,
    #     #                                             "size": eps_wall/4,
    #     #                                         },
    #     #                                 {
    #     #                                                             "center": (rotate(r_x+c_r[0], c_r[1], alpha)[0], rotate(r_x+c_r[0], c_r[1], alpha)[1]),
    #     #                                                             "radius": 0.5*r_x,
    #     #                                                             "thickness": 0.25,
    #     #                                                             "size": eps_wall/4,
    #     #                                                         }

    # ]

    # # ==========================================================
    # # 1) Raffinamento attorno al profilo
    # # ==========================================================

    # distance = gmsh.model.mesh.field.add("Distance")

    # gmsh.model.mesh.field.setNumbers(
    #     distance,
    #     "CurvesList",
    #     [l.dim_tag[1] for l in circle_lines],
    # )

    # wall = gmsh.model.mesh.field.add("Threshold")

    # gmsh.model.mesh.field.setNumber(
    #     wall,
    #     "InField",
    #     distance,
    # )

    # gmsh.model.mesh.field.setNumber(
    #     wall,
    #     "SizeMin",
    #     eps_wall,
    # )

    # gmsh.model.mesh.field.setNumber(
    #     wall,
    #     "SizeMax",
    #     eps_out,
    # )

    # gmsh.model.mesh.field.setNumber(
    #     wall,
    #     "DistMin",
    #     0.0,
    # )

    # gmsh.model.mesh.field.setNumber(
    #     wall,
    #     "DistMax",
    #     R_wall,
    # )

    # fields = [wall]

    # # ==========================================================
    # # 2) Raffinamenti locali
    # # ==========================================================

    # for ref in local_refinements:

    #     x, y = ref["center"]

    #     ball = gmsh.model.mesh.field.add("Ball")

    #     gmsh.model.mesh.field.setNumber(ball, "XCenter", x)
    #     gmsh.model.mesh.field.setNumber(ball, "YCenter", y)
    #     gmsh.model.mesh.field.setNumber(ball, "ZCenter", 0.0)

    #     gmsh.model.mesh.field.setNumber(ball, "Radius", ref["radius"])

    #     gmsh.model.mesh.field.setNumber(ball, "VIn", ref["size"])
    #     gmsh.model.mesh.field.setNumber(ball, "VOut", eps_out)

    #     # Alcune versioni di Gmsh non supportano Thickness:
    #     try:
    #         gmsh.model.mesh.field.setNumber(
    #             ball,
    #             "Thickness",
    #             ref["thickness"],
    #         )
    #     except:
    #         pass

    #     fields.append(ball)

    # # ==========================================================
    # # 3) Campo finale
    # # ==========================================================

    # background = gmsh.model.mesh.field.add("Min")

    # gmsh.model.mesh.field.setNumbers(
    #     background,
    #     "FieldsList",
    #     fields,
    # )

    # gmsh.model.mesh.field.setAsBackgroundMesh(background)

    # ==========================================================
    # Disabilita i controlli automatici di Gmsh
    # ==========================================================

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)


    # Needed before raw gmsh calls (replaces the invalid model.synchronize())
    

    # Assign EXPLICIT physical tags so we know exactly which integer
    # corresponds to which boundary in the FEniCSx script.
    gmsh.model.addPhysicalGroup(2, [plane_surface.dim_tag[1]], tag=1, name="Volume")
    gmsh.model.addPhysicalGroup(
            1, [l.dim_tag[1] for l in circle_lines_out], tag=2, name="c"
        )
    gmsh.model.addPhysicalGroup(
        1, [l.dim_tag[1] for l in circle_lines], tag=3, name="c"
    )  # circle

    geometry.generate_mesh(dim=2)
    gmsh.write(mesh_file)

    gmsh.clear()
    geometry.__exit__()

    mesh = meshio.read(mesh_file)

    triangle = mesh.cells_dict["triangle"]
    line = mesh.cells_dict["line"]
    triangle_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]

    # mesh volume
    meshio.write(
        "mesh.xdmf",
        meshio.Mesh(points=mesh.points, cells=[("triangle", triangle)],
        cell_data={
            "name_to_read":[triangle_tags]
        })
    )

    # facet markers must be written as INTEGER cell data
    meshio.write(
        "mf.xdmf",
        meshio.Mesh(
            points=mesh.points,
            cells=[("line", line)],
            cell_data={"f": [mesh.cell_data_dict["gmsh:physical"]["line"]]},
        ),
    )


def naca00xx_half_thickness(x, t=0.12):
    """Semispessore del NACA 00xx (x in [0,1], corda unitaria, bordo di fuga chiuso)."""
    return 5 * t * (0.2969 * np.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    - 0.1036 * x**4)


def generate_mesh_naca(L, H, a, chord=1.0, alpha=0.0, c_r=(0.0, 0.0),
                       t=0.12, n_pts=80, sizes=0.3, dists = None,
                       mesh_file="wing.msh"):
    """
    Mesh 2D rettangolare con profilo NACA0012.

    L, H      : lunghezza e altezza del dominio rettangolare
    a         : (0<a<1) il lato destro dista a*L dal centro dell'ala
                (quindi il lato sinistro dista (1-a)*L)
    chord     : lunghezza della corda
    alpha     : rotazione dell'ala [rad], antioraria, attorno al centro dell'ala
    c_r       : centro dell'ala (metà corda)
    t         : spessore relativo (0.12 -> NACA0012)
    n_pts     : punti per ciascuna superficie (dorso/ventre)
    size_wall : dimensione mesh sul profilo
    size_far  : dimensione mesh sul bordo esterno
    dist_max  : distanza dal profilo a cui la mesh raggiunge size_far
    """
    assert 0 < a < 1, "a deve essere compreso tra 0 e 1"
    if dists is None:
        dists = [min(H / 2, (1 - a) * L, a * L)]

    def rotate(x, y):
        """Ruota (x,y) di alpha attorno a c_r."""
        dx, dy = x - c_r[0], y - c_r[1]
        xr = c_r[0] + np.cos(alpha) * dx - np.sin(alpha) * dy
        yr = c_r[1] + np.sin(alpha) * dx + np.cos(alpha) * dy
        return (xr, yr, 0.0)

    # ------------------------------------------------------------------
    # Geometria del profilo
    # ------------------------------------------------------------------
    beta = np.linspace(0, np.pi, n_pts)
    xs = 0.5 * (1 - np.cos(beta))              # 0 -> 1, cluster ai bordi
    yt = naca00xx_half_thickness(xs, t)

    # coordinate fisiche: centro dell'ala a metà corda
    x_phys = c_r[0] + chord * (xs - 0.5)
    y_up = c_r[1] + chord * yt
    y_lo = c_r[1] - chord * yt

    # ------------------------------------------------------------------
    # Dominio rettangolare
    # ------------------------------------------------------------------
    x_min = c_r[0] - (1 - a) * L
    x_max = c_r[0] + a * L
    y_min = c_r[1] - H / 2
    y_max = c_r[1] + H / 2

    with pygmsh.geo.Geometry() as model:

        # --- bordo esterno ---
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        p_out = [model.add_point((x, y, 0.0), mesh_size=sizes[-1]) for x, y in corners]
        lines_out = [model.add_line(p_out[i], p_out[(i + 1) % 4]) for i in range(4)]

        # --- profilo ---
        p_le = model.add_point(rotate(x_phys[0], c_r[1]), mesh_size=sizes[0])
        p_te = model.add_point(rotate(x_phys[-1], c_r[1]), mesh_size=sizes[0])

        p_upper = [model.add_point(rotate(x_phys[i], y_up[i]), mesh_size=sizes[1])
                   for i in range(1, n_pts - 1)]
        p_lower = [model.add_point(rotate(x_phys[i], y_lo[i]), mesh_size=sizes[1])
                   for i in range(1, n_pts - 1)]

        # dorso: LE -> TE ; ventre: TE -> LE
        spline_up = model.add_spline([p_le] + p_upper + [p_te])
        spline_lo = model.add_spline([p_te] + p_lower[::-1] + [p_le])
        wing_lines = [spline_up, spline_lo]

        # --- superficie con buco ---
        loop_out = model.add_curve_loop(lines_out)
        loop_wing = model.add_curve_loop(wing_lines)
        surface = model.add_plane_surface(loop_out, holes=[loop_wing])

        model.synchronize()

        # --- raffinamento attorno al profilo ---
        # Distanza dal profilo alare
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist, "CurvesList", [l.dim_tag[1] for l in wing_lines]
        )
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)


        # Prima fascia: da 0 a d1
        f_thr1 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr1, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr1, "SizeMin", sizes[0])
        gmsh.model.mesh.field.setNumber(f_thr1, "SizeMax", sizes[1])
        gmsh.model.mesh.field.setNumber(f_thr1, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thr1, "DistMax", dists[0])
        gmsh.model.mesh.field.setNumber(f_thr1, "StopAtDistMax", 1)


        # Seconda fascia: da d1 a d2
        f_thr2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr2, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr2, "SizeMin", sizes[1])
        gmsh.model.mesh.field.setNumber(f_thr2, "SizeMax", sizes[2])
        gmsh.model.mesh.field.setNumber(f_thr2, "DistMin", dists[0])
        gmsh.model.mesh.field.setNumber(f_thr2, "DistMax", dists[1])


        # Combinazione: prende la dimensione minima
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_min, "FieldsList", [f_thr1, f_thr2]
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
        # f_dist = gmsh.model.mesh.field.add("Distance")
        # gmsh.model.mesh.field.setNumbers(
        #     f_dist, "CurvesList", [l.dim_tag[1] for l in wing_lines])
        # gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

        # f_thr = gmsh.model.mesh.field.add("Threshold")
        # gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        # gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", size_wall)
        # gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", size_far)
        # gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)
        # gmsh.model.mesh.field.setNumber(f_thr, "DistMax", dist_max)

        # gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)

        # la dimensione è governata solo dal campo
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # --- tag fisici espliciti ---
        gmsh.model.addPhysicalGroup(2, [surface.dim_tag[1]], tag=1, name="Volume")
        gmsh.model.addPhysicalGroup(1, [lines_out[1].dim_tag[1]], tag=2, name="inlet")
        gmsh.model.addPhysicalGroup(1, [lines_out[3].dim_tag[1]], tag=3, name="outlet")
        gmsh.model.addPhysicalGroup(1, [lines_out[0].dim_tag[1], lines_out[2].dim_tag[1]], tag=4, name="lateral")
        gmsh.model.addPhysicalGroup(1, [l.dim_tag[1] for l in wing_lines], tag=5, name="wing")



        model.generate_mesh(dim=2)
        gmsh.write(mesh_file)

    # ------------------------------------------------------------------
    # Conversione in XDMF
    # ------------------------------------------------------------------
    mesh = meshio.read(mesh_file)

    triangle = mesh.cells_dict["triangle"]
    line = mesh.cells_dict["line"]
    triangle_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]

    meshio.write(
        "mesh.xdmf",
        meshio.Mesh(points=mesh.points, cells=[("triangle", triangle)],
                    cell_data={"name_to_read": [triangle_tags]}),
    )
    meshio.write(
        "mf.xdmf",
        meshio.Mesh(points=mesh.points, cells=[("line", line)],
                    cell_data={"f": [line_tags]}),
    )


def generate_mesh_circular_naca(
    R,
    N2,
    chord=1.0,
    alpha=0.0,
    c_r=(0.0, 0.0),
    t=0.12,
    n_pts=100,
    sizes=(0.01, 0.3, 0.5),
    dists=(1.0, 2.0),
    mesh_file="wing.msh",
):
    """
    Mesh 2D con dominio circolare esterno e profilo NACA00xx interno.

    R       : raggio del dominio circolare
    N2      : numero di segmenti del bordo circolare
    chord   : corda del profilo
    alpha   : angolo di rotazione [rad]
    c_r     : centro della corda
    t       : spessore relativo NACA (0.12 = NACA0012)
    n_pts   : punti per ciascuna superficie del profilo
    sizes   : [size_wall, size_intermediate, size_far]
    dists   : [distanza prima fascia, distanza seconda fascia]
    """

    assert R > chord / 2, "Il profilo deve stare dentro il cerchio"
    assert N2 >= 8
    assert len(sizes) == 3
    assert len(dists) == 2
    assert 0 < dists[0] < dists[1]

    # ----------------------------------------------------------
    # Rotazione attorno al centro della corda
    # ----------------------------------------------------------

    def rotate(x, y):
        dx = x - c_r[0]
        dy = y - c_r[1]

        xr = c_r[0] + np.cos(alpha)*dx - np.sin(alpha)*dy
        yr = c_r[1] + np.sin(alpha)*dx + np.cos(alpha)*dy

        return (xr, yr, 0.0)

    # ----------------------------------------------------------
    # Parametri mesh
    # ----------------------------------------------------------

    eps_wall = sizes[0]
    eps_intermediate = sizes[1]
    eps_far = sizes[2]

    # ----------------------------------------------------------
    # Geometria NACA00xx
    # ----------------------------------------------------------

    beta = np.linspace(0, np.pi, n_pts)
    xs = 0.5 * (1 - np.cos(beta))

    yt = naca00xx_half_thickness(xs, t)

    x_phys = c_r[0] + chord * (xs - 0.5)

    y_up = c_r[1] + chord * yt
    y_lo = c_r[1] - chord * yt

    # ----------------------------------------------------------
    # Inizializzazione geometria
    # ----------------------------------------------------------

    with pygmsh.geo.Geometry() as model:

        # ======================================================
        # 1. DOMINIO CIRCOLARE ESTERNO
        # ======================================================

        circle_points = [
            model.add_point(
                (
                    c_r[0] + R*np.cos(2*np.pi*i/N2),
                    c_r[1] + R*np.sin(2*np.pi*i/N2),
                    0.0
                ),
                mesh_size=eps_far
            )
            for i in range(N2)
        ]

        circle_lines = [
            model.add_line(
                circle_points[i],
                circle_points[(i+1) % N2]
            )
            for i in range(N2)
        ]

        # ======================================================
        # 2. PROFILO NACA
        # ======================================================

        p_le = model.add_point(
            rotate(x_phys[0], c_r[1]),
            mesh_size=eps_wall
        )

        p_te = model.add_point(
            rotate(x_phys[-1], c_r[1]),
            mesh_size=eps_wall
        )

        p_upper = [
            model.add_point(
                rotate(x_phys[i], y_up[i]),
                mesh_size=eps_wall
            )
            for i in range(1, n_pts-1)
        ]

        p_lower = [
            model.add_point(
                rotate(x_phys[i], y_lo[i]),
                mesh_size=eps_wall
            )
            for i in range(1, n_pts-1)
        ]

        # Dorso: LE -> TE
        spline_up = model.add_spline(
            [p_le] + p_upper + [p_te]
        )

        # Ventre: TE -> LE
        spline_lo = model.add_spline(
            [p_te] + p_lower[::-1] + [p_le]
        )

        wing_lines = [spline_up, spline_lo]

        # ======================================================
        # 3. SUPERFICIE FLUIDA CON BUCO
        # ======================================================

        circle_loop = model.add_curve_loop(circle_lines)

        wing_loop = model.add_curve_loop(wing_lines)

        surface = model.add_plane_surface(
            circle_loop,
            holes=[wing_loop]
        )

        model.synchronize()

        # ======================================================
        # 4. RAFFINAMENTO ATTORNO AL PROFILO
        # ======================================================

        f_dist = gmsh.model.mesh.field.add("Distance")

        gmsh.model.mesh.field.setNumbers(
            f_dist,
            "CurvesList",
            [l.dim_tag[1] for l in wing_lines]
        )

        gmsh.model.mesh.field.setNumber(
            f_dist,
            "Sampling",
            200
        )

        # ------------------------------------------------------
        # Prima fascia: parete -> d1
        # ------------------------------------------------------

        f_thr1 = gmsh.model.mesh.field.add("Threshold")

        gmsh.model.mesh.field.setNumber(
            f_thr1, "InField", f_dist
        )

        gmsh.model.mesh.field.setNumber(
            f_thr1, "SizeMin", eps_wall
        )

        gmsh.model.mesh.field.setNumber(
            f_thr1, "SizeMax", eps_intermediate
        )

        gmsh.model.mesh.field.setNumber(
            f_thr1, "DistMin", 0.0
        )

        gmsh.model.mesh.field.setNumber(
            f_thr1, "DistMax", dists[0]
        )

        # ------------------------------------------------------
        # Seconda fascia: d1 -> d2
        # ------------------------------------------------------

        f_thr2 = gmsh.model.mesh.field.add("Threshold")

        gmsh.model.mesh.field.setNumber(
            f_thr2, "InField", f_dist
        )

        gmsh.model.mesh.field.setNumber(
            f_thr2, "SizeMin", eps_intermediate
        )

        gmsh.model.mesh.field.setNumber(
            f_thr2, "SizeMax", eps_far
        )

        gmsh.model.mesh.field.setNumber(
            f_thr2, "DistMin", dists[0]
        )

        gmsh.model.mesh.field.setNumber(
            f_thr2, "DistMax", dists[1]
        )

        # ------------------------------------------------------
        # Campo finale
        # ------------------------------------------------------

        f_min = gmsh.model.mesh.field.add("Min")

        gmsh.model.mesh.field.setNumbers(
            f_min,
            "FieldsList",
            [f_thr1, f_thr2]
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

        # La dimensione è governata dai campi
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints", 0
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", 0
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary", 0
        )

        # ======================================================
        # 5. PHYSICAL GROUPS
        # ======================================================

        gmsh.model.addPhysicalGroup(
            2, [surface.dim_tag[1]], tag=1
        )

        gmsh.model.setPhysicalName(
            2, 1, "Volume"
        )

        gmsh.model.addPhysicalGroup(
            1,
            [l.dim_tag[1] for l in circle_lines],
            tag=2
        )

        gmsh.model.setPhysicalName(
            1, 2, "ExternalBoundary"
        )

        gmsh.model.addPhysicalGroup(
            1,
            [l.dim_tag[1] for l in wing_lines],
            tag=3
        )

        gmsh.model.setPhysicalName(
            1, 3, "Wing"
        )

        # ======================================================
        # 6. GENERAZIONE MESH
        # ======================================================

        model.generate_mesh(dim=2)

        gmsh.write(mesh_file)

    # ==========================================================
    # 7. CONVERSIONE IN XDMF
    # ==========================================================

    mesh = meshio.read(mesh_file)

    triangle = mesh.cells_dict["triangle"]
    line = mesh.cells_dict["line"]

    triangle_tags = mesh.cell_data_dict[
        "gmsh:physical"
    ]["triangle"]

    line_tags = mesh.cell_data_dict[
        "gmsh:physical"
    ]["line"]

    meshio.write(
        "mesh.xdmf",
        meshio.Mesh(
            points=mesh.points,
            cells=[("triangle", triangle)],
            cell_data={
                "name_to_read": [triangle_tags]
            }
        )
    )

    meshio.write(
        "mf.xdmf",
        meshio.Mesh(
            points=mesh.points,
            cells=[("line", line)],
            cell_data={
                "f": [line_tags]
            }
        )
    )

    print("Mesh circolare NACA generata correttamente!")



if __name__ == "__main__":

    R = 3.0
    chord = 1.0

    c_r = (0.0, 0.0)

    N2 = 200
    alpha = np.pi 

    resolution = 0.01

    #generate_mesh_circular(L, r_x, r_y, N, N2, alpha, c_r, resolution)
    #generate_mesh_naca(L, H, a=0.5, chord=2*r_x, alpha=alpha, sizes=[0.01, 0.3, 0.5], dists=[1, 2], c_r=c_r)
    generate_mesh_circular_naca(
        R=R,
        N2=N2,
        chord=chord,
        alpha=alpha,
        c_r=c_r,
        t=0.12,
        n_pts=120,
        sizes=[resolution, 0.3, 0.5],
        dists=[1.0, 2.0],
        mesh_file="wing.msh"
    )

