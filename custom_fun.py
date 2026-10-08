import numpy as np
from mpi4py import MPI
from scipy.integrate import cumulative_trapezoid
from petsc4py import PETSc
import basix.ufl
from basix.ufl import element, mixed_element

from dolfinx import fem, io
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import ufl
from ufl import as_tensor, indices, nabla_grad, Identity, grad, inner, dx, dot, div

import sys
sys.path.append("/home/ferra/wing_dynamics/modules")
import matplotlib.pyplot as plt
import plot as pltmd
import mesh_generation as mg
from estrazione_spline import crea_spline_estesa, crea_funzioni

def custom_sin(T, num_steps):
    x = np.linspace(0, T, num_steps)
    f1_t = lambda t: -2*np.pi*40/180*2*np.pi*np.sin(2*2*np.pi*(t))
    f2_t = lambda t: -np.pi*40/180*2*np.pi*(np.sin(2*2*np.pi*(t-0.25*(1-np.exp(-2*t))))*(1-0.5*np.exp(-4*t)))
    f3_t = lambda t: -2*np.pi*40/180*2*np.pi*2*np.pi*np.cos(2*2*np.pi*t)

    F1 = cumulative_trapezoid([f1_t(x_) for x_ in x], x, initial=0)
    F2 = cumulative_trapezoid([f2_t(x_) for x_ in x], x, initial=0)

    plt.plot(x, (F1-(max(F1)+min(F1))/2), label="F1(x)")
    plt.plot(x, (F2-(max(F2)+min(F2))/2), label="F2(x)")

    plt.legend()
    plt.show()

    return f1_t, f2_t, f3_t, x

from scipy.interpolate import CubicSpline

def custom_di(T, num_steps, num_steps_=1000,  period_phi=0.5, amp_phi=np.pi*40/180,
              period_alpha=0.5, amp_alpha=np.pi*40/180,
              sigma_phi=0.02, sigma_alpha=0.02, phase_alpha=0.72, n_grid=4096):
    """
    phi(t)   : onda triangolare a media nulla, ampiezza di picco amp_phi, periodo period_phi
    alpha(t) : onda rettangolare a media nulla, ampiezza amp_alpha, periodo period_alpha
    Lo smoothing è una convoluzione (periodica) del segnale ideale con un kernel gaussiano
    di deviazione standard sigma_* (in secondi). Velocità e accelerazione si ottengono
    convolvendo con le derivate analitiche del kernel: d(s*g)/dt = s*g'.

    sigma_phi, sigma_alpha: default 5% del rispettivo periodo (phi) e 3% (alpha).
    phase_alpha: sfasamento di alpha come frazione del periodo di alpha.

    Ritorna: f1_t (dphi/dt), f2_t (dalpha/dt), f3_t (d2phi/dt2), x
    """
    x = np.linspace(0, T, num_steps_)

    if sigma_phi is None:
        sigma_phi = 0.05*period_phi
    if sigma_alpha is None:
        sigma_alpha = 0.03*period_alpha

    def _smooth_periodic(ideal, period, sigma, n_der_max):
        """
        ideal: funzione del tempo (array) -> segnale ideale periodico.
        Ritorna lista di spline periodiche [s, s', s'', ...] fino a n_der_max.
        """
        N = n_grid
        dt = period/N
        t = np.arange(N)*dt
        s = ideal(t)
        s = s - s.mean()                                  # media nulla
        
        # kernel gaussiano e derivate analitiche, centrato in 0 (circolare)
        tk = (np.arange(N) - N//2)*dt
        g = np.exp(-0.5*(tk/sigma)**2)
        g /= g.sum()*dt                                   # normalizzazione: integrale = 1
        kernels = [g,
                   -tk/sigma**2*g,
                   (tk**2/sigma**4 - 1/sigma**2)*g]

        S = np.fft.rfft(s)
        splines = []
        for n in range(n_der_max + 1):
            K = np.fft.rfft(np.fft.ifftshift(kernels[n]))
            y = np.fft.irfft(S*K, n=N)*dt                 # convoluzione circolare
            y_ext = np.append(y, y[0])                    # chiusura periodica
            t_ext = np.append(t, period)
            splines.append(CubicSpline(t_ext, y_ext, bc_type="periodic", extrapolate="periodic"))
        return splines

    # segnali ideali
    tri = lambda t: amp_phi*(2/np.pi)*np.arcsin(np.sin(2*np.pi*t/period_phi))  # picco +-amp_phi, aggiungo esponenziale iniziale
    sq  = lambda t: amp_alpha*np.where(np.sin(2*np.pi*(t/period_alpha - phase_alpha)) >= 0, 1.0, -1.0)

    phi_s, dphi_s, d2phi_s = _smooth_periodic(tri, period_phi,   sigma_phi,   2)
    al_s,  dal_s           = _smooth_periodic(sq,  period_alpha, sigma_alpha, 1)

    tau = period_phi / 5

    def w(t):
        return 1.0 - np.exp(-(t / tau)**2)

    def dw(t):
        return 2.0 * t / tau**2 * np.exp(-(t / tau)**2)

    def d2w(t):
        return (
            2.0 / tau**2
            - 4.0 * t**2 / tau**4
        ) * np.exp(-(t / tau)**2)


    phi_t = lambda t: (
        np.atleast_1d(phi_s(t)) * w(t)
    )

    alpha_t = lambda t: (
        np.atleast_1d(al_s(t))
    )

    f1_t = lambda t: (
        np.atleast_1d(dphi_s(t)) * w(t)
        + np.atleast_1d(phi_s(t)) * dw(t)
    )

    f2_t = lambda t: (
        np.atleast_1d(dal_s(t))
    )

    f3_t = lambda t: (
        np.atleast_1d(d2phi_s(t)) * w(t)
        + 2.0 * np.atleast_1d(dphi_s(t)) * dw(t)
        + np.atleast_1d(phi_s(t)) * d2w(t)
    )

    # --- Plot di controllo
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(x, phi_t(x), label="phi(t) smooth")
    ax[0].plot(x, alpha_t(x), label="alpha(t) smooth")
    ax[0].plot(x, tri(x) - 0, ":", color="gray", label="ideali")
    ax[0].plot(x, sq(x), ":", color="gray")
    ax[0].legend(); ax[0].set_ylabel("posizioni")

    ax[1].plot(x, f1_t(x), label="f1 = dphi/dt")
    ax[1].plot(x, f2_t(x), label="f2 = dalpha/dt")
    ax[1].plot(x, f3_t(x), label="f3 = d2phi/dt2")
    ax[1].legend(); ax[1].set_xlabel("t"); ax[1].set_ylabel("derivate")
    plt.tight_layout()
    plt.show()

    # check: integrando f1 e f2 si devono ritrovare phi e alpha
    F1 = cumulative_trapezoid(f1_t(x), x, initial=0)
    F2 = cumulative_trapezoid(f2_t(x), x, initial=0)
    plt.plot(x, F1 - F1.mean(), label="int f1 (media tolta)")
    plt.plot(x, phi_t(x) - phi_t(x).mean(), "--", label="phi")
    plt.plot(x, F2 - F2.mean(), label="int f2 (media tolta)")
    plt.plot(x, alpha_t(x) - alpha_t(x).mean(), "--", label="alpha")
    plt.legend()
    plt.show()

    return f1_t, f2_t, f3_t, x

def zanzara(T, num_steps, t_0=0.25, t_1=0.5):
    x = np.hstack([np.linspace(0, 0.8, num_steps)-t, np.linspace(0.81, T, num_steps*int(3))-t])
    x = np.linspace(0, T, num_steps)
    funzioni = crea_funzioni(
        "dati.txt",
        n_armoniche=5
    )

    f1_t = lambda t: funzioni["fourier"]["dpositional_dt"](t)
    f2_t_ = lambda t: funzioni["fourier"]["dfeathering_tip_dt"](t)
    f3_t = lambda t: funzioni["fourier"]["d2positional_dt2"](t)

    F1 = lambda t: funzioni["fourier"]["positional"](t)
    F2 = lambda t: funzioni["fourier"]["feathering_tip"](t)

    # file_txt = "dati.txt" 
    # ( positional, d_positional_dt, d2_positional_dt2, elevation, d_elevation_dt, d2_elevation_dt2, feathering_tip, d_feathering_dt, d2_feathering_dt2 ) = crea_spline_estesa( file_txt, t_min=-1, t_max=4.0 )


    # f1_t = d_positional_dt
    # f2_t = d_feathering_dt
    # f3_t = d2_positional_dt2

    # F1 = positional
    # F2 = feathering_tip



    # plt.plot(x, (F1-(max(F1)+min(F1))/2), label="F1(x)")
    # plt.plot(x, (F2-(max(F2)+min(F2))/2), label="F2(x)")

    # x_esteso = x_esteso[ordine] 
    # x_esteso_feathering = x_esteso - x_0 * np.exp(-2*(x_esteso-x_0))
    # f2_t = lambda t: f2_t_(t - t_0 * np.exp(-2 * (t-t_0)))*(1-2* t_0 * np.exp(-2 * (t-t_0)))*(1-np.exp(-2 * (t-t_1))) + F2(t - t_0 * np.exp(-2 * (t-t_0)))** 2 * np.exp(-2 * (t-t_1))
    # f1_t = lambda t: f1_t_(t)*(1-np.exp(-2 * (t-t_1))) + F1(t)* 2 * np.exp(-2 * (t-t_1))
    # f3_t = lambda t: f3_t_(t)*(1-np.exp(-2 * (t-t_1))) + 4*f1_t_(t)* np.exp(-2 * (t-t_1)) - F1(t)* 4 * np.exp(-2 * (t-t_1))ù
    f2_t = lambda t: f2_t_(t - t_0 * np.exp(-2 * (t-t_0)))*(1+2* t_0 * np.exp(-2 * (t-t_0)))
    #F3 = cumulative_trapezoid(f2_t(x), x, initial=0)
    plt.plot(x, F1(x), label="F1(x)")
    plt.plot(x, F2(x), label="F2(x)")
    #plt.plot(x, (F3-(max(F3)+min(F3))/2), label="F3(x)", marker=".", linestyle="None")
    #plt.plot(x[0:-1], [(F2(x[i+1])-F2(x[i]))/(x[i+1]-x[i]) for i in range(x.shape[0]-1)], label="f2_t(x)", marker=".", linestyle="None")
    plt.plot(x[0:-1], [sum([f2_t(x[j])*(x[j+1]-x[j]) for j in range(i)])+1.34 for i in range(x.shape[0]-1)], label="f2_t(x)", marker=".", linestyle="None")
    plt.plot(x, f1_t(x), label="V1(x)")
    plt.plot(x, f2_t(x), label="V2(x)")
    plt.plot(x, f3_t(x), label="A1(x)")

    plt.legend()
    plt.show()

    return f1_t, f2_t, f3_t, x