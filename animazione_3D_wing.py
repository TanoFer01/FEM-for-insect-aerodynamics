#!/usr/bin/env python3
"""
Animazione interattiva di un'ala (faccia rettangolare) che segue una traiettoria
definita da tre angoli piu' l'inclinazione tra due sistemi di riferimento.

Figura con 3 pannelli sincronizzati:
  1) vista 3D dell'ala (con sfera di raggio 1 opzionale: flag SHOW_SPHERE)
  2) theta(t), phi(t), alpha(t) in tre grafici impilati (finestra scorrevole)
  3) traiettoria nel piano (phi, theta) con il segmento della corda
     che forma l'angolo alpha con la tangente alla traiettoria

Convenzioni
-----------
* S  : sistema fisso (x, y, z).
* S' : S ruotato di BETA attorno all'asse x.
* theta, phi : angoli sferici (colatitudine, azimut) in S' del versore r(t)
       (asse dell'ala):  r = (sin th cos ph, sin th sin ph, cos th)
* v(t) = dr/dt nel piano tangente alla sfera.
* ALPHA : angolo, nel piano tangente, tra corda e v:
           corda = cos(a) v_hat + sin(a) (r_hat x v_hat)

Uso:  python ala_traiettoria.py      (richiede numpy e matplotlib)
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------------------
# FLAG / opzioni di visualizzazione
# ----------------------------------------------------------------------------
SHOW_SPHERE = True     # True: mostra la sfera di raggio 1 nel grafico 3D
SEG_MODE = "plot"      # vettori nel piano (phi,theta):
                       #   "plot"   -> angoli esatti *visivamente* nel grafico
                       #               (costruiti in pixel, quindi non distorti dagli assi)
                       #   "sfera"  -> proiezione fedele dei vettori 3D in coord. (phi, theta)
SEG_PX = 45.0          # semi-lunghezza del segmento della corda nel piano (phi,theta) [pixel]
T_WIN = 8.0            # finestra temporale mostrata nei grafici [s]

# ----------------------------------------------------------------------------
# Parametri iniziali (gradi dove indicato)
# ----------------------------------------------------------------------------
P = dict(
    beta=0.0,         # inclinazione S -> S' [deg]
    theta0=0.0,      # theta medio [deg]
    dtheta=10.0,      # ampiezza oscillazione di theta [deg]
    f_theta=0.5,      # frequenza oscillazione di theta [Hz]
    omega_phi=80.0,   # ampiezza/velocita' di phi [deg]
    dt=0.0,           # ritardo [s]
    phi0=0.0,         # phi iniziale [deg]
    alpha=-50.0,      # ampiezza angolo ala-velocita' [deg]
    speed=1.0,        # moltiplicatore del tempo
    CD=0,           # coefficiente di resistenza
    CL=0,           # coefficiente di portanza
)
VEL_SCALE = 0.3   # lunghezza 3D della freccia velocita' per |v| = 1
F_SCALE = 0.1     # lunghezza 3D della freccia forza per |F| = 1
VPX = 25.0        # grafico 3: pixel per |v| = 1   (modo "plot")
FPX = 10.0        # grafico 3: pixel per |F| = 1   (modo "plot")
PLOT_VEC = 0.5    # grafico 3, modo "sfera": fattore sulle frecce 3D
L = 1.0        # lunghezza dell'ala (apertura) = raggio della sfera
W = 0.45       # larghezza (corda) dell'ala
DT = 0.03      # passo temporale base per frame [s]
TRAIL_LEN = 400


# ----------------------------------------------------------------------------
# Geometria / funzioni del moto  (angoli in gradi)
# ----------------------------------------------------------------------------
def Rx(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def theta_fn(t):
    return P["theta0"] + P["dtheta"] * np.sin(2 * np.pi * P["f_theta"] * t)


def phi_fn(t):
    return P["phi0"] + P["omega_phi"] * np.cos(np.pi * P["f_theta"] * (t - P["dt"]))


def alpha_fn(t):
    return P["alpha"] * np.sin(np.pi * P["f_theta"] * t)


def r_prime(t):
    """Versore asse ala in S' al tempo t."""
    th, ph = np.radians(theta_fn(t)), np.radians(phi_fn(t))
    return np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])


def wing_frame(t):
    """Restituisce (r_hat, v_hat, chord_hat, |v|) in S' al tempo t."""
    h = 1e-4
    r = r_prime(t)
    v = (r_prime(t + h) - r_prime(t - h)) / (2 * h)
    v = v - np.dot(v, r) * r                    # proiezione nel piano tangente
    n = np.linalg.norm(v)
    if n < 1e-9:                                # velocita' nulla: direzione arbitraria
        tmp = np.array([0, 0, 1.0]) if abs(r[2]) < 0.9 else np.array([1.0, 0, 0])
        v = np.cross(tmp, r)
        n = np.linalg.norm(v)
    v_hat = v / n
    a = np.radians(alpha_fn(t))
    chord = np.cos(a) * v_hat + np.sin(a) * np.cross(r, v_hat)
    return r, v_hat, chord / np.linalg.norm(chord), n


def sph_proj(vec, th_deg, ph_deg):
    """Vettore 3D (in S') -> offset (d_phi, d_theta) in gradi nel piano (phi, theta)."""
    th, ph = np.radians(th_deg), np.radians(ph_deg)
    e_th = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    e_ph = np.array([-np.sin(ph), np.cos(ph), 0.0])
    s_ = max(abs(np.sin(th)), 1e-3)
    return np.degrees([np.dot(vec, e_ph) / s_, np.dot(vec, e_th)])


# ----------------------------------------------------------------------------
# Figura e layout
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8.5))

# --- 1) 3D --------------------------------------------------------------
ax = fig.add_axes([0.0, 0.25, 0.54, 0.73], projection="3d")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
#ax.set_title("Ala su traiettoria (beta, phi, theta, alpha)")
ax.view_init(elev=25, azim=-60)

# sfera di raggio 1 (flag SHOW_SPHERE); invariante per rotazione, quindi uguale in S e S'
if SHOW_SPHERE:
    u = np.linspace(0, 2 * np.pi, 37)
    w = np.linspace(0, np.pi, 19)
    xs = np.outer(np.cos(u), np.sin(w))
    ys = np.outer(np.sin(u), np.sin(w))
    zs = np.outer(np.ones_like(u), np.cos(w))
    ax.plot_wireframe(xs, ys, zs, color="steelblue", alpha=0.25, linewidth=0.4)

# assi S (grigio) e S' (colorati)
for vec, lab in zip(np.eye(3), "xyz"):
    ax.plot(*zip([0, 0, 0], 1.3 * vec), color="gray", lw=1)
    ax.text(*(1.38 * vec), lab, color="gray")
sp_lines, sp_texts = [], []
for col in ("r", "g", "b"):
    sp_lines.append(ax.plot([0, 0], [0, 0], [0, 0], color=col, lw=1.5)[0])
    sp_texts.append(ax.text(0, 0, 0, "", color=col))

plane_sp = Poly3DCollection([np.zeros((4, 3))], alpha=0.12, facecolor="tab:orange", edgecolor="tab:orange")
ax.add_collection3d(plane_sp)
plane_s_corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
ax.add_collection3d(Poly3DCollection([plane_s_corners], alpha=0.06, facecolor="gray", edgecolor="gray"))

# ala, vettori, scia
wing = Poly3DCollection([np.zeros((4, 3))], alpha=0.75, facecolor="tab:blue", edgecolor="k")
ax.add_collection3d(wing)
axis_line, = ax.plot([0, 0], [0, 0], [0, 0], color="k", lw=2)
vel_line, = ax.plot([0, 0], [0, 0], [0, 0], color="red", lw=2.5)
chord_line, = ax.plot([0, 0], [0, 0], [0, 0], color="green", lw=2.5)
force_line, = ax.plot([0, 0], [0, 0], [0, 0], color="magenta", lw=2.5)
trail_line, = ax.plot([], [], [], color="tab:purple", lw=1, alpha=0.8)
tip_pt, = ax.plot([0], [0], [0], "ko", ms=4)
info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, family="monospace")
ax.plot([], [], color="k", lw=2, label="asse ala r")
ax.plot([], [], color="red", lw=2, label="velocita' v")
ax.plot([], [], color="green", lw=2, label="corda")
#ax.plot([], [], color="magenta", lw=2, label="F = -CD|v|^2 u_v + CL|v|^2 u_z")
ax.legend(loc="upper right")

# --- 2) theta(t), phi(t), alpha(t) ---------------------------------------
RX, RW, RH = 0.62, 0.35, 0.125
ax_th = fig.add_axes([RX, 0.84, RW, RH])
ax_ph = fig.add_axes([RX, 0.68, RW, RH], sharex=ax_th)
ax_al = fig.add_axes([RX, 0.52, RW, RH], sharex=ax_th)
ts_axes = [ax_th, ax_ph, ax_al]
ts_fns = [theta_fn, phi_fn, alpha_fn]
ts_cols = ["tab:blue", "tab:orange", "tab:green"]
ts_labels = ["theta [deg]", "phi [deg]", "alpha [deg]"]
ts_lines, ts_pts = [], []
for a_, col, lab in zip(ts_axes, ts_cols, ts_labels):
    a_.set_ylabel(lab)
    a_.grid(alpha=0.3)
    ln, = a_.plot([], [], color=col, lw=1.6)
    pt, = a_.plot([], [], "o", color=col, ms=6)
    ts_lines.append(ln); ts_pts.append(pt)
ax_th.tick_params(labelbottom=False)
ax_ph.tick_params(labelbottom=False)
ax_al.set_xlabel("t [s]")
ax_th.set_title("Evoluzione temporale")

# --- 3) traiettoria nel piano (phi, theta) -------------------------------
ax_pt = fig.add_axes([RX, 0.11, RW, 0.28])
ax_pt.set_xlabel("phi [deg]"); ax_pt.set_ylabel("theta [deg]")
ax_pt.set_title("Traiettoria (phi, theta) e corda")
ax_pt.grid(alpha=0.3)
pt_traj, = ax_pt.plot([], [], color="tab:purple", lw=1.2, alpha=0.8)
pt_chord, = ax_pt.plot([], [], color="green", lw=3, label="corda")
pt_pos, = ax_pt.plot([], [], "ko", ms=5)
ax_pt.plot([], [], color="red", lw=2, label="v (ampiezza ~ |v|)")
#ax_pt.plot([], [], color="magenta", lw=2, label="F")
_arrow = dict(arrowstyle="->", lw=2, shrinkA=0, shrinkB=0)
ann_v = ax_pt.annotate("", xy=(0, 0), xytext=(0, 0), annotation_clip=False,
                       arrowprops=dict(color="red", **_arrow))
# ann_f = ax_pt.annotate("", xy=(0, 0), xytext=(0, 0), annotation_clip=False,
#                        arrowprops=dict(color="magenta", **_arrow))
alpha_txt = ax_pt.text(0.02, 0.9, "", transform=ax_pt.transAxes, family="monospace", fontsize=9)
ax_pt.legend(loc="upper right", fontsize=7)

# ----------------------------------------------------------------------------
# Stato e funzioni di disegno
# ----------------------------------------------------------------------------
trail = deque(maxlen=TRAIL_LEN)
state = dict(t=0.0, running=True)


def set_line(line, p0, p1):
    line.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])


def update_frames():
    """Ridisegna gli elementi che dipendono da beta (sistema S' e suo piano xy)."""
    R = Rx(np.radians(P["beta"]))
    for i, (line, txt, lab) in enumerate(zip(sp_lines, sp_texts, ("x'", "y'", "z'"))):
        e = R[:, i]
        set_line(line, [0, 0, 0], 1.3 * e)
        txt.set_position((1.38 * e[0], 1.38 * e[1])); txt.set_3d_properties(1.38 * e[2])
        txt.set_text(lab)
    plane_sp.set_verts([(plane_s_corners @ R.T)])


def pad_lim(c, amp, minamp, k=1.25):
    a = max(abs(amp), minamp) * k
    return c - a, c + a


def draw(t):
    R = Rx(np.radians(P["beta"]))
    r_p, v_p, c_p, speed = wing_frame(t)     # in S'
    # direzione "u_z": perpendicolare a v nel piano (v, corda), verso z di S positivo
    n_p = np.cross(r_p, v_p)
    sgn = 1.0 if (R @ n_p)[2] >= 0 else -1.0
    uz_p = sgn * n_p
    F_p = speed ** 2 * (-P["CD"] * v_p + P["CL"] * uz_p)       # forza in S'
    r, v, c, F = R @ r_p, R @ v_p, R @ c_p, R @ F_p            # S' -> S
    o = np.zeros(3)
    tip = L * r

    # --- 1) 3D
    wing.set_verts([[o - W / 2 * c, o + W / 2 * c, tip + W / 2 * c, tip - W / 2 * c]])
    set_line(axis_line, o, 1.1 * L * r)
    set_line(vel_line, tip, tip + VEL_SCALE * speed * v)       # ampiezza ~ |v|
    set_line(chord_line, tip, tip + 0.4 * c)
    set_line(force_line, tip, tip + F_SCALE * F)
    trail.append(tip)
    tr = np.array(trail)
    trail_line.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
    tip_pt.set_data_3d([tip[0]], [tip[1]], [tip[2]])
    th_now, ph_now, al_now = theta_fn(t), phi_fn(t), alpha_fn(t)
    # info.set_text(f"t={t:6.2f}s  theta={th_now:6.1f}  phi={ph_now:6.1f}  "
    #               f"beta={P['beta']:.0f}  alpha={al_now:6.1f}  |v|={speed:5.2f}")

    # --- finestra temporale (passato)
    tt = np.linspace(t - T_WIN, t, 400)

    # --- 2) serie temporali
    ts_amps = [(P["theta0"], P["dtheta"], 5), (P["phi0"], P["omega_phi"], 5), (0.0, P["alpha"], 5)]
    for a_, ln, pt, fn, (c0, amp, mn) in zip(ts_axes, ts_lines, ts_pts, ts_fns, ts_amps):
        ln.set_data(tt, fn(tt))
        pt.set_data([t], [fn(t)])
        a_.set_ylim(*pad_lim(c0, amp, mn))
    ax_th.set_xlim(t - T_WIN, t + 0.05 * T_WIN)

    # --- 3) piano (phi, theta)
    pt_traj.set_data(phi_fn(tt), theta_fn(tt))
    ax_pt.set_xlim(*pad_lim(P["phi0"], abs(P["omega_phi"]) + 25, 5, 1.1))
    ax_pt.set_ylim(*pad_lim(P["theta0"], abs(P["dtheta"]) + 10, 5, 1.1))
    P0 = np.array([ph_now, th_now])

    if SEG_MODE == "plot":
        # tutto costruito in coordinate di schermo (pixel): gli angoli sono quelli visti
        h = 1e-3
        tv = np.array([(phi_fn(t + h) - phi_fn(t - h)) / (2 * h),
                       (theta_fn(t + h) - theta_fn(t - h)) / (2 * h)])
        tv = tv / np.linalg.norm(tv) if np.linalg.norm(tv) > 1e-9 else np.array([1.0, 0.0])
        tr_ = ax_pt.transData
        d0 = tr_.transform(P0)
        T = tr_.transform(P0 + tv) - d0
        T = T / np.linalg.norm(T)                   # tangente (pixel)
        N = np.array([T[1], -T[0]])                 # ruotato di -90 deg (verso di r x v) #@G HO CAMBIATO IL SEGNO DI TUTTO IL VETTORE
        a = np.radians(-al_now)
        chord_dir = np.cos(a) * T + np.sin(a) * N   # angolo alpha esatto con la tangente
        inv = tr_.inverted()
        p_a = inv.transform(d0 - SEG_PX * chord_dir)
        p_b = inv.transform(d0 + SEG_PX * chord_dir)
        v_end = inv.transform(d0 + VPX * speed * T)
        F_end = inv.transform(d0 + FPX * speed ** 2 * (-P["CD"] * T - P["CL"] * sgn * N))
    else:
        h_ = np.degrees(W / 2)
        p_a = P0 - sph_proj(0.5 * W * c_p, th_now, ph_now) * 1.0
        p_b = P0 + sph_proj(0.5 * W * c_p, th_now, ph_now) * 1.0
        v_end = P0 + PLOT_VEC * sph_proj(VEL_SCALE * speed * v_p, th_now, ph_now)
        F_end = P0 + PLOT_VEC * sph_proj(F_SCALE * F_p, th_now, ph_now)

    pt_chord.set_data([p_a[0], p_b[0]], [p_a[1], p_b[1]])
    pt_pos.set_data([ph_now], [th_now])
    ann_v.xy = tuple(v_end); ann_v.set_position(tuple(P0))
    #ann_f.xy = tuple(F_end); ann_f.set_position(tuple(P0))
    #alpha_txt.set_text(f"alpha = {al_now:6.1f} deg\n|v| = {speed:5.2f}")


def step(_):
    if state["running"]:
        state["t"] += DT * P["speed"]
    draw(state["t"])


# ----------------------------------------------------------------------------
# Controlli interattivi
# ----------------------------------------------------------------------------
# camera
elev_slider = Slider(fig.add_axes([0.10, 0.025, 0.17, 0.02]), "Elevation", -90, 90, valinit=25)
azim_slider = Slider(fig.add_axes([0.35, 0.025, 0.17, 0.02]), "Azimuth", -180, 180, valinit=-60)


def update_view(val):
    ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
    fig.canvas.draw_idle()


elev_slider.on_changed(update_view)
azim_slider.on_changed(update_view)

specs = [
    ("beta", "beta [deg]", -90, 90),
    ("alpha", "A_alpha [deg]", -90, 90),
    ("dtheta", "A_theta ", 0, 85),
    ("omega_phi", "A_phi [deg]", -180, 180),
    ("f_theta", "f theta [Hz]", 0, 2),
    #("dt", "ritardo [s]", 0, 0.5),
    ("speed", "velocita'", 0.1, 3),
    #("CD", "C_D", 0, 2),
    #("CL", "C_L", 0, 2),
]
sliders = []
for i, (key, label, lo, hi) in enumerate(specs):
    col, row = i % 2, i // 2
    sax = fig.add_axes([0.10 + col * 0.25, 0.215 - row * 0.036, 0.17, 0.022])
    s = Slider(sax, label, lo, hi, valinit=P[key])

    def on_change(val, key=key):
        P[key] = val
        if key == "beta":
            update_frames()
        trail.clear()

    s.on_changed(on_change)
    sliders.append(s)

btn_pause = Button(fig.add_axes([RX, 0.01, 0.1, 0.04]), "Pausa")
btn_reset = Button(fig.add_axes([RX + 0.12, 0.01, 0.1, 0.04]), "Reset scia")


def toggle(_):
    state["running"] = not state["running"]
    btn_pause.label.set_text("Riprendi" if not state["running"] else "Pausa")


btn_pause.on_clicked(toggle)
btn_reset.on_clicked(lambda _: trail.clear())

update_frames()
anim = FuncAnimation(fig, step, interval=30, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()