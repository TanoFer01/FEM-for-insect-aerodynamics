import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline


def leggi_dati(file_txt):
    """
    Legge il file TXT.

    Colonne:
        0 -> Stroke cycle
        1 -> Positional angle
        2 -> Elevation angle
        3 -> Feathering angle tip
    """

    dati = []

    with open(file_txt, "r", encoding="utf-8") as f:

        for riga in f:

            riga = riga.strip()

            if not riga:
                continue

            valori = riga.split()

            try:
                valori = [
                    float(v.replace(",", "."))
                    for v in valori
                ]
            except ValueError:
                continue

            if len(valori) >= 4:
                dati.append(valori[:4])

    if len(dati) < 4:
        raise ValueError(
            "Non sono stati trovati abbastanza dati numerici."
        )

    dati = np.asarray(dati)

    x = dati[:, 0]
    positional = dati[:, 1]
    elevation = dati[:, 2]
    feathering = dati[:, 3]

    # Ordina per Stroke cycle
    ordine = np.argsort(x)

    x = x[ordine]
    positional = positional[ordine]
    elevation = elevation[ordine]
    feathering = feathering[ordine]

    # Elimina eventuali duplicati
    x, indici = np.unique(
        x,
        return_index=True
    )

    positional = positional[indici]
    elevation = elevation[indici]
    feathering = feathering[indici]

    return x, positional, elevation, feathering


def crea_spline_estesa(
    file_txt,
    t_min=-0.5,
    t_max=4.0
):
    """
    Crea spline cubiche estese replicando periodicamente
    il ciclo originale.

    La spline finale è definita nell'intervallo:

        [t_min, t_max]

    Per default:

        [-0.5, 4.0]

    Restituisce tre funzioni:

        positional(t)
        elevation(t)
        feathering_tip(t)

    Le funzioni NON fanno modulo di t.
    Sono vere spline definite sull'intero intervallo esteso.
    """

    # =========================================================
    # LETTURA DATI
    # =========================================================

    (
        x,
        positional,
        elevation,
        feathering
    ) = leggi_dati(file_txt)

    positional *= 2*np.pi/360
    elevation *= 2*np.pi/360
    feathering *= 2*np.pi/360
    # =========================================================
    # CONTROLLI
    # =========================================================

    if x[0] < 0 or x[-1] > 1:
        raise ValueError(
            "Lo Stroke cycle deve essere compreso "
            "nell'intervallo 0-1."
        )

    # =========================================================
    # COSTRUZIONE DEI CICLI
    # =========================================================

    # Cicli necessari per coprire [-0.5, 4]
    cicli = range(
        int(np.floor(t_min)) - 1,
        int(np.ceil(t_max)) + 1
    )

    x_esteso = []
    positional_esteso = []
    elevation_esteso = []
    feathering_esteso = []

    for ciclo in cicli:

        # Traslazione del ciclo originale
        x_ciclo = x + ciclo 

        # Evita il punto finale duplicato tra due cicli.
        #
        # Esempio:
        #
        # ciclo 0 -> arriva a 1
        # ciclo 1 -> parte da 1
        #
        # teniamo solo uno dei due.

        if ciclo != min(cicli):
            mask = x_ciclo > ciclo
        else:
            mask = np.ones_like(
                x_ciclo,
                dtype=bool
            )

        x_esteso.extend(
            x_ciclo[mask]
        )

        positional_esteso.extend(
            positional[mask]
        )

        elevation_esteso.extend(
            elevation[mask]
        )

        feathering_esteso.extend(
            feathering[mask]
        )

    x_esteso = np.asarray(x_esteso)
    positional_esteso = np.asarray(
        positional_esteso
    )
    elevation_esteso = np.asarray(
        elevation_esteso
    )
    feathering_esteso = np.asarray(
        feathering_esteso
    )

    # =========================================================
    # ORDINA
    # =========================================================

    ordine = np.argsort(x_esteso)

    x_0 = 0.14
    x_esteso = x_esteso[ordine] 
    x_esteso_feathering = x_esteso - x_0 * np.exp(-2*(x_esteso-x_0))
    #print(x_esteso)

    positional_esteso = \
        positional_esteso[ordine]

    elevation_esteso = \
        elevation_esteso[ordine]

    feathering_esteso = \
        feathering_esteso[ordine]

    # =========================================================
    # LIMITA ESATTAMENTE ALL'INTERVALLO RICHIESTO
    # =========================================================

    mask = (
        (x_esteso >= t_min) &
        (x_esteso <= t_max)
    )

    x_esteso = x_esteso[mask]

    x_esteso_feathering = x_esteso_feathering[mask]

    positional_esteso = \
        positional_esteso[mask]

    elevation_esteso = \
        elevation_esteso[mask]

    feathering_esteso = \
        feathering_esteso[mask]

    # =========================================================
    # SPLINE
    # =========================================================

    spline_positional = UnivariateSpline(x_esteso, positional_esteso, s=0.01)
    # CubicSpline(
    #     x_esteso,
    #     positional_esteso
    # )

    spline_elevation = UnivariateSpline(x_esteso, elevation_esteso, s=0.01)
    # CubicSpline(
    #     x_esteso,
    #     elevation_esteso
    # )

    spline_feathering = UnivariateSpline(x_esteso_feathering, feathering_esteso, s=0.01)
    # CubicSpline(
    #     x_esteso_feathering,
    #     feathering_esteso
    # )

    # =========================================================
    # FUNZIONI DA RESTITUIRE
    # =========================================================

    def positional_fun(t):
        return spline_positional(t)

    def elevation_fun(t):
        return spline_elevation(t)

    def feathering_tip_fun(t):
        return spline_feathering(t)

    #Prima derivata 
    d_positional_dt = lambda t: spline_positional.derivative(1)(t)
    d_elevation_dt = lambda t: spline_elevation.derivative(1)(t)
    d_feathering_dt = lambda t: spline_feathering.derivative(1)(t)
    # Seconda derivata
    d2_positional_dt2 = lambda t: spline_positional.derivative(2)(t)
    d2_elevation_dt2 = lambda t: spline_elevation.derivative(2)(t)
    d2_feathering_dt2 = lambda t: spline_feathering.derivative(2)(t)
    # ========================================================= # FUNZIONI DA RESTITUIRE # ========================================================= def positional(t): return spline_positional(t) def d_positional_dt(t): return d_spline_positional(t) def d2_positional_dt2(t): return d2_spline_positional(t) def elevation(t): return spline_elevation(t) def d_elevation_dt(t): return d_spline_elevation(t) def d2_elevation_dt2(t): return d2_spline_elevation(t) def feathering_tip(t): return spline_feathering(t) def d_feathering_dt(t): return d_spline_feathering(t) def d2_feathering_dt2(t): return d2_spline_feathering(t) return ( positional, d_positional_dt, d2_positional_dt2, elevation, d_elevation_dt, d2_elevation_dt2, feathering_tip, d_feathering_dt, d2_feathering_dt2 )
    return ( positional_fun, d_positional_dt, d2_positional_dt2, elevation_fun, d_elevation_dt, d2_elevation_dt2, feathering_tip_fun, d_feathering_dt, d2_feathering_dt2 )

def main():

    file_txt = "dati.txt"

    # =========================================================
    # CREA LE SPLINE
    # =========================================================

    (
        positional,
        elevation,
        feathering_tip
    ) = crea_spline_estesa(
        file_txt,
        t_min=-0.5,
        t_max=4.0
    )

    # =========================================================
    # PLOT
    # =========================================================

    t = np.linspace(
        -0.5,
        4.0,
        5000
    )

    plt.figure(figsize=(12, 8))

    plt.plot(
        t,
        positional(t),
        label="Positional angle"
    )

    plt.plot(
        t,
        elevation(t),
        label="Elevation angle"
    )

    plt.plot(
        t,
        feathering_tip(t),
        label="Feathering angle tip"
    )

    # Evidenzia i cicli
    for ciclo in range(0, 5):

        plt.axvline(
            ciclo,
            linestyle="--",
            alpha=0.4
        )

    plt.xlabel("Stroke cycle")
    plt.ylabel("Angle [deg]")

    plt.title(
        "Spline cubica estesa da -0.5 a 4"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ============================================================
# LETTURA DEL FILE
# ============================================================

def leggi_dati(file_txt):
    """
    Legge il file TXT.

    Colonne utilizzate:
        0 -> Stroke cycle
        1 -> Positional angle
        2 -> Elevation angle
        3 -> Feathering angle (tip)

    Gestisce la virgola come separatore decimale.
    """

    dati = []

    with open(file_txt, "r", encoding="utf-8") as f:

        for riga in f:

            riga = riga.strip()

            if not riga:
                continue

            valori = riga.split()

            try:
                valori = [
                    float(v.replace(",", "."))
                    for v in valori
                ]
            except ValueError:
                continue

            if len(valori) >= 4:
                dati.append(valori)

    if len(dati) < 4:
        raise ValueError(
            "Non sono stati trovati abbastanza dati numerici."
        )

    dati = np.asarray(dati)

    x = dati[:, 0]
    positional = dati[:, 1]
    elevation = dati[:, 2]
    feathering = -dati[:, 3]

    # Ordina rispetto a Stroke cycle
    ordine = np.argsort(x)

    x = x[ordine]
    positional = positional[ordine]*2*np.pi/360
    elevation = elevation[ordine]*2*np.pi/360
    feathering = feathering[ordine]*2*np.pi/360

    # Elimina eventuali duplicati
    x_unique, indici = np.unique(
        x,
        return_index=True
    )

    x = x_unique
    positional = positional[indici]
    elevation = elevation[indici]
    feathering = feathering[indici]

    return (
        x,
        positional,
        elevation,
        feathering
    )


# ============================================================
# SPLINE CUBICA
# ============================================================

def crea_spline(x, y):
    """
    Crea una spline cubica sui dati.
    """

    return CubicSpline(x, y)


# ============================================================
# FIT DI FOURIER
# ============================================================

def crea_fourier(x, y, n_armoniche=10):
    """
    Crea un fit periodico di Fourier.

    f(t) = a0 +
           sum[
               ak*cos(2*pi*k*t)
               + bk*sin(2*pi*k*t)
           ]

    Parametri
    ---------
    x : array
        Stroke cycle [0,1]

    y : array
        Valori da interpolare/fittare

    n_armoniche : int
        Numero di armoniche Fourier.

    Returns
    -------
    f
        Funzione periodica

    df
        Prima derivata

    d2f
        Seconda derivata
    """

    periodo = 1.0

    # ---------------------------------------------------------
    # Normalizzazione di x
    # ---------------------------------------------------------

    x = np.asarray(x)
    y = np.asarray(y)

    x = np.mod(x, periodo)

    # Se abbiamo sia 0 che 1, 1 diventa 0.
    # Per il fit Fourier sono lo stesso punto.
    x_unique, indici = np.unique(
        x,
        return_index=True
    )

    x = x_unique
    y = y[indici]

    # ---------------------------------------------------------
    # MATRICE DI FOURIER
    # ---------------------------------------------------------

    A = np.ones(
        (len(x), 1 + 2 * n_armoniche)
    )

    for k in range(1, n_armoniche + 1):

        A[:, 2 * k - 1] = np.cos(
            2.0 * np.pi * k * x
        )

        A[:, 2 * k] = np.sin(
            2.0 * np.pi * k * x
        )

    # ---------------------------------------------------------
    # LEAST SQUARES
    # ---------------------------------------------------------

    coefficienti, _, _, _ = np.linalg.lstsq(
        A,
        y,
        rcond=None
    )

    a0 = coefficienti[0]

    ak = coefficienti[1::2]
    bk = coefficienti[2::2]

    # ---------------------------------------------------------
    # FUNZIONE
    # ---------------------------------------------------------

    def f(t):

        t = np.asarray(t)

        risultato = np.full_like(
            t,
            a0,
            dtype=float
        )

        for k in range(1, n_armoniche + 1):

            omega_k = 2.0 * np.pi * k

            risultato += (
                ak[k - 1] *
                np.cos(omega_k * t)
                +
                bk[k - 1] *
                np.sin(omega_k * t)
            )

        return risultato

    # ---------------------------------------------------------
    # PRIMA DERIVATA
    # ---------------------------------------------------------

    def df(t):

        t = np.asarray(t)

        risultato = np.zeros_like(
            t,
            dtype=float
        )

        for k in range(1, n_armoniche + 1):

            omega_k = 2.0 * np.pi * k

            risultato += (
                -ak[k - 1] *
                omega_k *
                np.sin(omega_k * t)
                +
                bk[k - 1] *
                omega_k *
                np.cos(omega_k * t)
            )

        return risultato

    # ---------------------------------------------------------
    # SECONDA DERIVATA
    # ---------------------------------------------------------

    def d2f(t):

        t = np.asarray(t)

        risultato = np.zeros_like(
            t,
            dtype=float
        )

        for k in range(1, n_armoniche + 1):

            omega_k = 2.0 * np.pi * k

            risultato += (
                -ak[k - 1] *
                omega_k**2 *
                np.cos(omega_k * t)
                -
                bk[k - 1] *
                omega_k**2 *
                np.sin(omega_k * t)
            )

        return risultato

    return f, df, d2f


# ============================================================
# CREA TUTTE LE FUNZIONI
# ============================================================

def crea_funzioni(file_txt, n_armoniche=10):
    """
    Legge il file e crea le funzioni periodiche.

    Returns
    -------

    spline:
        tuple contenente le tre spline

    fourier:
        tuple contenente:
            funzioni
            prime derivate
            seconde derivate
    """

    (
        x,
        positional,
        elevation,
        feathering
    ) = leggi_dati(file_txt)

    # ---------------------------------------------------------
    # SPLINE
    # ---------------------------------------------------------

    spline_positional = crea_spline(
        x,
        positional
    )

    spline_elevation = crea_spline(
        x,
        elevation
    )

    spline_feathering = crea_spline(
        x,
        feathering
    )

    # ---------------------------------------------------------
    # FOURIER
    # ---------------------------------------------------------

    (
        fourier_positional,
        dfourier_positional,
        d2fourier_positional
    ) = crea_fourier(
        x,
        positional,
        n_armoniche
    )

    (
        fourier_elevation,
        dfourier_elevation,
        d2fourier_elevation
    ) = crea_fourier(
        x,
        elevation,
        n_armoniche
    )

    (
        fourier_feathering,
        dfourier_feathering,
        d2fourier_feathering
    ) = crea_fourier(
        x,
        feathering,
        n_armoniche
    )

    # ---------------------------------------------------------
    # FUNZIONI PERIODICHE
    # ---------------------------------------------------------

    def positional(t):
        return spline_positional(
            np.mod(t, 1.0)
        )

    def elevation(t):
        return spline_elevation(
            np.mod(t, 1.0)
        )

    def feathering_tip(t):
        return spline_feathering(
            np.mod(t, 1.0)
        )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    return {

        "spline": {
            "positional": positional,
            "elevation": elevation,
            "feathering_tip": feathering_tip
        },

        "fourier": {
            "positional": fourier_positional,
            "elevation": fourier_elevation,
            "feathering_tip": fourier_feathering,

            "dpositional_dt":
                dfourier_positional,

            "delevation_dt":
                dfourier_elevation,

            "dfeathering_tip_dt":
                dfourier_feathering,

            "d2positional_dt2":
                d2fourier_positional,

            "d2elevation_dt2":
                d2fourier_elevation,

            "d2feathering_tip_dt2":
                d2fourier_feathering
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    file_txt = "dati.txt"

    # Numero di armoniche Fourier
    n_armoniche = 10

    funzioni = crea_funzioni(
        file_txt,
        n_armoniche=n_armoniche
    )

    # ---------------------------------------------------------
    # FUNZIONI FOURIER
    # ---------------------------------------------------------

    positional = \
        funzioni["fourier"]["positional"]

    elevation = \
        funzioni["fourier"]["elevation"]

    feathering = \
        funzioni["fourier"]["feathering_tip"]

    dpositional = \
        funzioni["fourier"]["dpositional_dt"]

    delevation = \
        funzioni["fourier"]["delevation_dt"]

    dfeathering = \
        funzioni["fourier"]["dfeathering_tip_dt"]

    # ---------------------------------------------------------
    # TEMPO
    # ---------------------------------------------------------

    t = np.linspace(
        0.0,
        3.0,
        3000
    )

    # =========================================================
    # GRAFICO FUNZIONI
    # =========================================================

    plt.figure(figsize=(12, 8))

    plt.plot(
        t,
        positional(t),
        label="Positional angle"
    )

    plt.plot(
        t,
        elevation(t),
        label="Elevation angle"
    )

    plt.plot(
        t,
        feathering(t),
        label="Feathering angle tip"
    )

    plt.axvline(
        1.0,
        linestyle="--",
        alpha=0.5
    )

    plt.axvline(
        2.0,
        linestyle="--",
        alpha=0.5
    )

    plt.xlabel("Stroke cycle")
    plt.ylabel("Angle [deg]")

    plt.title(
        f"Fit Fourier - {n_armoniche} armoniche"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # =========================================================
    # GRAFICO DERIVATE
    # =========================================================

    plt.figure(figsize=(12, 8))

    plt.plot(
        t,
        dpositional(t),
        label="d Positional / dt"
    )

    plt.plot(
        t,
        delevation(t),
        label="d Elevation / dt"
    )

    plt.plot(
        t,
        dfeathering(t),
        label="d Feathering / dt"
    )

    plt.axvline(
        1.0,
        linestyle="--",
        alpha=0.5
    )

    plt.axvline(
        2.0,
        linestyle="--",
        alpha=0.5
    )

    plt.xlabel("Stroke cycle")
    plt.ylabel("dAngle / dt")

    plt.title(
        "Derivate temporali"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# ESECUZIONE DIRETTA
# ============================================================


if __name__ == "__main__":
    main()

