#!/usr/bin/env python3
"""
Animazione interattiva di un'ala (faccia rettangolare) che segue una traiettoria
definita da tre angoli piu' l'inclinazione tra due sistemi di riferimento.

Convenzioni
-----------
* S  : sistema fisso (x, y, z).
* S' : sistema ottenuto ruotando S di un angolo BETA attorno all'asse x
       (quindi il piano xy di S' e' inclinato di BETA rispetto a quello di S).
* theta, phi : angoli sferici (colatitudine, azimut) misurati in S' che
       definiscono il versore r(t) = asse dell'ala (asse di apertura alare):
           r = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
* Il piano tangente alla sfera nel punto r e' perpendicolare all'asse dell'ala,
  quindi la sezione dell'ala (la corda) giace nel piano tangente.
* v(t) = dr/dt : velocita' istantanea della punta del vettore (nel piano tangente).
* ALPHA : angolo, misurato nel piano tangente, tra la corda dell'ala e v.
           corda = cos(alpha) * v_hat + sin(alpha) * (r_hat x v_hat)
  La faccia rettangolare dell'ala e' il piano generato da r_hat (apertura)
  e dalla corda; l'ala e' incernierata nell'origine.

Traiettoria (modificabile in theta_t / phi_t):
    phi(t)   = phi0 + omega_phi * t
    theta(t) = theta0 + dtheta * sin(2*pi*f_theta*t)

Uso:  python ala_traiettoria.py      (richiede numpy e matplotlib)
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------------------
# Parametri iniziali (gradi dove indicato)
# ----------------------------------------------------------------------------
P = dict(
    beta=0.0,        # inclinazione S -> S' [deg]
    theta0=90.0,      # theta medio [deg]
    dtheta=10.0,      # ampiezza oscillazione di theta [deg]
    f_theta=0.5,      # frequenza oscillazione di theta [Hz]
    omega_phi=80.0,   # velocita' angolare di phi [deg/s]
    dt = 0.0,          # ritardo ... [s]
    phi0=0.0,         # phi iniziale [deg]
    alpha=-50.0,       # angolo ala-velocita' nel piano tangente [deg]
    speed=1.0,        # moltiplicatore del tempo
)
L = 1.0        # lunghezza dell'ala (apertura) = raggio della sfera
W = 0.15       # larghezza (corda) dell'ala
DT = 0.03      # passo temporale base per frame [s]
TRAIL_LEN = 400


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------
def Rx(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def r_prime(t):
    """Versore asse ala in S' al tempo t."""
    th = np.radians(P["theta0"] - P["dtheta"] * np.sin(2 * np.pi * P["f_theta"] * (t) ))
    ph = np.radians(P["phi0"] + P["omega_phi"] * np.cos( np.pi * P["f_theta"] * (t-P["dt"])))
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def wing_frame(t):
    """Restituisce (r_hat, v_hat, chord_hat) in S' al tempo t."""
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
    a = np.sin( np.pi * P["f_theta"] *  t) * np.radians(P["alpha"])
    #a = np.radians(P["alpha"])
    chord = np.cos(a) * v_hat + np.sin(a) * np.cross(r, v_hat)
    return r, v_hat, chord / np.linalg.norm(chord)


# ----------------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.02, 0.28, 0.96, 0.70], projection="3d")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Ala su traiettoria (beta, phi, theta, alpha)")

# sfera di riferimento (wireframe leggero, in S')
u = np.linspace(0, 2 * np.pi, 25)
w = np.linspace(0, np.pi, 13)
xs = np.outer(np.cos(u), np.sin(w)); ys = np.outer(np.sin(u), np.sin(w)); zs = np.outer(np.ones_like(u), np.cos(w))
sph = np.stack([xs, ys, zs], axis=-1) @ np.eye(3)
sphere_artist = [None]

# assi S (grigio) e S' (colorati) -- S' viene ridisegnato quando cambia beta
for vec, lab in zip(np.eye(3), "xyz"):
    ax.plot(*zip([0, 0, 0], 1.3 * vec), color="gray", lw=1)
    ax.text(*(1.38 * vec), lab, color="gray")
sp_lines, sp_texts = [], []
for col in ("r", "g", "b"):
    sp_lines.append(ax.plot([0, 0], [0, 0], [0, 0], color=col, lw=1.5)[0])
    sp_texts.append(ax.text(0, 0, 0, "", color=col))

# piano xy di S' (semi-trasparente) e di S (contorno)
plane_sp = Poly3DCollection([np.zeros((4, 3))], alpha=0.12, facecolor="tab:orange", edgecolor="tab:orange")
ax.add_collection3d(plane_sp)
plane_s_corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
ax.add_collection3d(Poly3DCollection([plane_s_corners], alpha=0.06, facecolor="gray", edgecolor="gray"))

# Camera elevation slider
elev_ax = fig.add_axes([0.14, 0.025, 0.28, 0.02])
elev_slider = Slider(
    elev_ax, "Elevation", -90, 90, valinit=0
)

# Camera azimuth slider
azim_ax = fig.add_axes([0.55, 0.025, 0.28, 0.02])
azim_slider = Slider(
    azim_ax, "Azimuth", -180, 180, valinit=0
)

def update_view(val):
    ax.view_init(
        elev=elev_slider.val,
        azim=azim_slider.val
    )
    fig.canvas.draw_idle()

elev_slider.on_changed(update_view)
azim_slider.on_changed(update_view)

# ala, vettori, scia
wing = Poly3DCollection([np.zeros((4, 3))], alpha=0.75, facecolor="tab:blue", edgecolor="k")
ax.add_collection3d(wing)
axis_line, = ax.plot([0, 0], [0, 0], [0, 0], color="k", lw=2)
vel_line, = ax.plot([0, 0], [0, 0], [0, 0], color="red", lw=2.5)
chord_line, = ax.plot([0, 0], [0, 0], [0, 0], color="green", lw=2.5)
trail_line, = ax.plot([], [], [], color="tab:purple", lw=1, alpha=0.8)
tip_pt, = ax.plot([0], [0], [0], "ko", ms=4)
info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, family="monospace")
ax.plot([], [], color="k", lw=2, label="asse ala r")
ax.plot([], [], color="red", lw=2, label="velocita' v")
ax.plot([], [], color="green", lw=2, label="corda")
ax.legend(loc="upper right")


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


def draw(t):
    R = Rx(np.radians(P["beta"]))
    r, v, c = wing_frame(t)
    r, v, c = R @ r, R @ v, R @ c           # S' -> S
    o = np.zeros(3)
    tip = L * r
    wing.set_verts([[o - W / 2 * c, o + W / 2 * c, tip + W / 2 * c, tip - W / 2 * c]])
    set_line(axis_line, o, 1.1 * L * r)
    set_line(vel_line, tip, tip + 0.4 * v)
    set_line(chord_line, tip, tip + 0.4 * c)
    trail.append(tip)
    tr = np.array(trail)
    trail_line.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
    tip_pt.set_data_3d([tip[0]], [tip[1]], [tip[2]])
    rp = r_prime(t)
    th = np.degrees(np.arccos(np.clip(rp[2], -1, 1))); ph = np.degrees(np.arctan2(rp[1], rp[0])) % 360
    info.set_text(f"t={t:6.2f}s  theta={th:6.1f}  phi={ph:6.1f}  beta={P['beta']:.0f}  alpha={P['alpha']:.0f}")


def step(_):
    if state["running"]:
        state["t"] += DT * P["speed"]
    draw(state["t"])


# ----------------------------------------------------------------------------
# Controlli interattivi
# ----------------------------------------------------------------------------
specs = [
    ("beta", "beta [deg]", -90, 90),
    ("alpha", "alpha [deg]", -90, 90),
    ("dtheta", "ampiezza theta", 0, 85),
    ("omega_phi", "omega phi [deg/s]", -180, 180),
    ("f_theta", "f theta [Hz]", 0, 2),
    ("dt", "ritardo [s]", 0, 0.5),
    ("speed", "velocita'", 0.1, 3),
]
sliders = []
for i, (key, label, lo, hi) in enumerate(specs):
    col, row = i % 2, i // 2
    sax = fig.add_axes([0.14 + col * 0.45, 0.20 - row * 0.045, 0.28, 0.025])
    s = Slider(sax, label, lo, hi, valinit=P[key])

    def on_change(val, key=key):
        P[key] = val
        if key in ("beta",):
            update_frames()
        trail.clear()

    s.on_changed(on_change)
    sliders.append(s)

bax = fig.add_axes([0.59, 0.065, 0.1, 0.04])
btn_pause = Button(bax, "Pausa")
rax = fig.add_axes([0.71, 0.065, 0.1, 0.04])
btn_reset = Button(rax, "Reset scia")


def toggle(_):
    state["running"] = not state["running"]
    btn_pause.label.set_text("Riprendi" if not state["running"] else "Pausa")


btn_pause.on_clicked(toggle)
btn_reset.on_clicked(lambda _: trail.clear())

update_frames()
anim = FuncAnimation(fig, step, interval=30, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()