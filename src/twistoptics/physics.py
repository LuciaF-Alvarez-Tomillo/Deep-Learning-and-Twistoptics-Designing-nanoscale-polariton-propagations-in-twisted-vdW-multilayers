# -*- coding: utf-8 -*-
"""
===========================================================
IFC_Definitions
===========================================================
This script contains the analytical isofrequency functions and their
parameters used in the document.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import numpy as np
from functools import partial
from iminuit import Minuit
from scipy.signal import find_peaks
from twistoptics.materials import (
    eps_SiO2_exp,
    eps_Au,
    eps_CoTeMoO6,
    eps_V2O5_4parOsc,
    eps_MoO3_exp_IR_THz_2,
    eps_MoO3_exp_IR_THz_3,
    eps_MoOCl2,
)
# =============================================================================
# Rp condition, if false, analitical calculation of the IFC is done.
# =============================================================================

Rp = False

# =============================================================================
# Material and Substrate Mapping
# =============================================================================
MATERIAL_FUNCTIONS = {
    "MoO3": eps_MoO3_exp_IR_THz_2,
    "MoO3THz": eps_MoO3_exp_IR_THz_3,
    "V2O5": eps_V2O5_4parOsc,
    "MoOCl2": eps_MoOCl2,
    "CoTeMoO6": eps_CoTeMoO6,
}

SUBSTRATE_FUNCTIONS = {
    "SiO2": eps_SiO2_exp,
    "Au": eps_Au,
}

# =============================================================================
# R
# =============================================================================


def Ctes_Perm(params):
    """
    Calculate system constants and permittivities for multilayer structure.
    Converts input parameters to physical constants including permittivity tensors,
    wave vectors, and layer properties.
    """
    # System parameters
    d_layers_nm = params[0]
    angles_deg = params[1]
    wavelength_mu = params[2]
    eps_superstrate = params[3]
    mat_substrate = params[4]
    mat = params[5]

    # Unit conversions
    d_layers = d_layers_nm * 1e-9  # Convert nanometers to meters

    # Convert angles from degrees to radians
    angles_rad = np.array([0, angles_deg[0], angles_deg[1]]) * np.pi / 180

    # Calculate wavelength and wave vector
    wl = wavelength_mu * 1e-6  # Convert micrometers to meters
    omega = 1 / wl * 1e-2
    k0 = 2.0 * np.pi / wl

    # Substrate permittivity
    if mat_substrate not in SUBSTRATE_FUNCTIONS:
        raise ValueError(f"No valid substrate material: {mat_substrate}")
    eps_substrate = SUBSTRATE_FUNCTIONS[mat_substrate](omega)
    E1 = eps_substrate
    E2 = eps_superstrate

    # Calculate layer permittivity tensor components
    Ex, Ey, Ez = [], [], []
    for material in mat:
        if material not in MATERIAL_FUNCTIONS:
            raise ValueError(f"No valid material: {material}")
        func = MATERIAL_FUNCTIONS[material]
        Ex.append(func(omega, "A"))
        Ey.append(func(omega, "B"))
        Ez.append(func(omega, "C"))

    # Convert to arrays
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    Ez = np.array(Ez)

    # Calculate complex permittivity for isotropic media
    Esub = 1j * E1
    Esup = 1j * E2

    ctes = [
        k0,
        Ex,
        Ey,
        Ez,
        angles_rad[0],
        angles_rad[1],
        angles_rad[2],
        d_layers[0],
        d_layers[1],
        d_layers[2],
        Esub,
        Esup,
    ]
    return ctes


def C_shiz_trilayer(ctes, theta_real):
    """
    Calculate phase shifts and reflection coefficients for trilayer structure.
    Computes qz values and C coefficients from equation 3 of supplementary S1.
    """
    ang = theta_real

    k0 = ctes[0]
    Ex = ctes[1]
    Ey = ctes[2]
    Ez = ctes[3]
    Esub = ctes[10]
    Esup = ctes[11]
    a11 = ctes[4]
    a12 = ctes[5]
    a13 = ctes[6]
    d1 = ctes[7]
    d2 = ctes[8]
    d3 = ctes[9]

    E1x = Ex[0]
    E2x = Ex[1]
    E3x = Ex[2]
    E1y = Ey[0]
    E2y = Ey[1]
    E3y = Ey[2]
    E1z = Ez[0]
    E2z = Ez[1]
    E3z = Ez[2]

    q1z = (
        -E1x / E1z * (np.cos(ang + a11)) ** 2 - E1y / E1z * (np.sin(ang + a11)) ** 2
    ) ** 0.5
    q2z = (
        -E2x / E2z * (np.cos(ang + a12)) ** 2 - E2y / E2z * (np.sin(ang + a12)) ** 2
    ) ** 0.5
    q3z = (
        -E3x / E3z * (np.cos(ang + a13)) ** 2 - E3y / E3z * (np.sin(ang + a13)) ** 2
    ) ** 0.5

    Eq1z = E1z * q1z
    Eq2z = E2z * q2z
    Eq3z = E3z * q3z
    Eq12z = Eq1z * Eq2z
    Eq13z = Eq1z * Eq3z
    Eq23z = Eq2z * Eq3z
    Eq123z = Eq12z * Eq3z

    shi1z = q1z * k0 * d1
    shi2z = q2z * k0 * d2
    shi3z = q3z * k0 * d3
    shiz = np.array([shi1z, shi2z, shi3z])

    C0 = Eq123z * (Esub + Esup)
    C1 = (-1j) * Eq23z * (Esub * Esup + Eq1z**2)
    C2 = (-1j) * Eq13z * (Esub * Esup + Eq2z**2)
    C3 = (-1j) * Eq12z * (Esub * Esup + Eq3z**2)
    C12 = (-1) * Eq3z * (Esub * Eq2z**2 + Esup * Eq1z**2)
    C13 = (-1) * Eq2z * (Esub * Eq3z**2 + Esup * Eq1z**2)
    C23 = (-1) * Eq1z * (Esub * Eq3z**2 + Esup * Eq2z**2)
    C123 = 1j * (Esub * Eq2z**2 * Esup + Eq13z**2)
    C = np.array([C0, C1, C2, C3, C12, C13, C23, C123])

    return shiz, C


def R_trilayer(shiz, C, q_real, q_imag):
    """
    Calculate reflection coefficient for trilayer structure.
    Computes the complex reflection coefficient using phase shifts and coefficients.
    """
    shi1z = shiz[0]
    shi2z = shiz[1]
    shi3z = shiz[2]

    C0 = C[0]
    C1 = C[1]
    C2 = C[2]
    C3 = C[3]
    C12 = C[4]
    C13 = C[5]
    C23 = C[6]
    C123 = C[7]

    q0 = q_real + 1j * q_imag

    qshi1z = shi1z * q0
    qshi2z = shi2z * q0
    qshi3z = shi3z * q0

    tan_qshi1z = np.tan(qshi1z)
    tan_qshi2z = np.tan(qshi2z)
    tan_qshi3z = np.tan(qshi3z)
    tan_qshi12z = tan_qshi1z * tan_qshi2z
    tan_qshi13z = tan_qshi1z * tan_qshi3z
    tan_qshi23z = tan_qshi2z * tan_qshi3z
    tan_qshi123z = tan_qshi12z * tan_qshi3z
    R = C0
    R = R + (tan_qshi1z) * C1
    R = R + (tan_qshi2z) * C2
    R = R + (tan_qshi3z) * C3
    R = R + (tan_qshi12z) * C12
    R = R + (tan_qshi13z) * C13
    R = R + (tan_qshi23z) * C23
    R = R + (tan_qshi123z) * C123
    R = np.cos(qshi1z) * np.cos(qshi2z) * np.cos(qshi3z) * R
    return R


def R_abs_trilayer(shiz, C, q_real, q_imag):
    """
    Calculate absolute value of reflection coefficient for trilayer structure.
    """
    R = R_trilayer(shiz, C, q_real, q_imag)
    return np.abs(R)


# =============================================================================
# IFC with Analytical Rp
# =============================================================================


def Fresnel_coeff_ctes(ang, ctes):
    # Extract constants required to evaluate the Fresnel coefficient
    # from the precomputed structure constants array (`ctes`).
    k0 = ctes[0]
    Ex = ctes[1]
    Ey = ctes[2]
    Ez = ctes[3]
    Esub = ctes[10] / 1j
    Esup = ctes[11] / 1j
    a11 = ctes[4]
    a12 = ctes[5]
    a13 = ctes[6]
    d1 = ctes[7]
    d2 = ctes[8]
    d3 = ctes[9]

    E1x = Ex[0]
    E2x = Ex[1]
    E3x = Ex[2]
    E1y = Ey[0]
    E2y = Ey[1]
    E3y = Ey[2]
    E1z = Ez[0]
    E2z = Ez[1]
    E3z = Ez[2]
    Eix = np.array([Esub, E1x, E2x, E3x, Esup])
    Eiy = np.array([Esub, E1y, E2y, E3y, Esup])
    Eiz = np.array([Esub, E1z, E2z, E3z, Esup])

    Per = np.column_stack([Eix, Eiy, Eiz])

    Per = np.asarray(Per, dtype=complex)

    D = [0, d1, d2, d3, 0]

    D = np.asarray(D, dtype=float)

    layers = [1, 2, 3]  # np.where(D > 0)[0]
    # N_layers = len(layers)
    N_meds = len(D)
    meds_3D = np.arange(N_meds)

    d = D[layers]  # Thicknesses of physical layers only (excluding super/substrate)

    Eix = Per[meds_3D, 0]
    Eiy = Per[meds_3D, 1]
    Eiz = Per[meds_3D, 2]

    als = [0, a11, a12, a13, 0]

    qiz = np.sqrt(
        (-Eix / Eiz) * (np.cos(ang + als)) ** 2
        + (-Eiy / Eiz) * (np.sin(ang + als)) ** 2
        + 0j
    )

    Eqiz = Eiz * qiz
    shiz = qiz[layers] * k0 * d

    return Eqiz, shiz, Eiz


def Fresnel_coeff(Eqiz, shiz, Eiz, q_real):
    """
    Calculate Fresnel reflection coefficient for p-polarized light.
    Uses recursive calculation through multilayer structure.
    """
    N_layers = 3

    q = q_real

    Eqiz = Eqiz * q

    shiz = shiz * q
    c_shiz = np.cos(shiz)
    s_shiz = np.sin(shiz)

    denom0 = Eqiz[1] * (-1j) * Eiz[0] * q * c_shiz[0] + 1j * (Eqiz[1]) ** 2 * s_shiz[0]

    if np.abs(denom0) < 1e-7:
        D_val = np.nan
    else:
        D_val = (Eqiz[1] * c_shiz[0] + Eiz[0] * q * s_shiz[0]) / (denom0)

    for j in range(1, N_layers):
        denom = Eqiz[j + 1] * c_shiz[j] + 1j * D_val * (Eqiz[j + 1]) ** 2 * s_shiz[j]
        if np.abs(denom) < 1e-7:
            D_val = np.nan
            break
        D_val = (Eqiz[j + 1] * D_val * c_shiz[j] + 1j * s_shiz[j]) / denom

    if np.isnan(D_val):
        Rp = np.nan
    else:
        denom = -1j * Eiz[-1] * D_val * q + 1
        if denom == 0:
            Rp = np.nan
        else:
            Rp = (1j * Eiz[-1] * D_val * q + 1) / denom

    return Rp


def RP_first_mode(qr0, ang, er):
    """
    Extract fundamental mode reflection coefficient for given angles.
    Finds the first peak in the error matrix for each angle (see Sup. S5).

    Parameters:
    -----------
    qr0 : array
        Array of radial wave vectors (Mr,)
    ang : array
        Array of angles (N_ang,)
    er : matrix
        Error matrix (Mr, N_ang)
    """
    rho = qr0
    phi = ang
    Z = er.T

    N_ang = len(phi)

    direction1 = range(len(phi))
    Mid_ang = N_ang // 2
    direction2 = list(range(Mid_ang, -1, -1)) + list(range(N_ang - 1, Mid_ang, -1))

    for r in range(2):
        if r == 0:
            direction = direction1
            index = -1  # compare with the previous angular sample
            qmode = np.zeros(N_ang)
        else:
            direction = direction2
            index = 1  # compare with the next angular sample
            qmode1 = qmode
            qmode = np.zeros(N_ang)

        for i in direction:
            signal = Z[:, i]
            if np.all(np.isnan(signal)):
                qmode[i] = 0
                continue

            peaks, props = find_peaks(signal, height=np.nanmax(signal) * 0.01)
            # 0.01 is the relative height threshold to consider a peak

            if len(peaks) > 0:
                valid_idx = (props["peak_heights"] > 0.1) & (qr0[peaks] >= 2)
                # 0.1 is the absolute height threshold to consider a valid peak (physical observable)
                # and qr0 >= 2 to avoid non-confined solutions

                if np.any(valid_idx):
                    qmode[i] = rho[peaks[valid_idx][0]]
                else:
                    qmode[i] = 0
            else:
                qmode[i] = 0

            if i > 0 and i < (N_ang - 1) and qmode[i] >= 2:
                if (qmode[i] - qmode[i + index]) > 3 and qmode[i + index] >= 2.1:
                    # If the current mode is significantly larger than the adjacent one, and the adjacent one is a valid mode, replace it with the adjacent one to ensure continuity.
                    qmode[i] = qmode[i + index]

                if (qmode[i] - qmode[i + index]) > 3 and abs(
                    qmode[i + index] - 2
                ) < 0.1:
                    # If the current mode is significantly larger than the adjacent one, and the adjacent one is close to 2 (non-confined), replace it with a small value to avoid non-physical jumps.
                    qmode[i] = 2

    qmode = np.minimum(qmode, qmode1)

    # Initial and final points correction to avoid non-physical jumps at the edges of the angular range
    if (qmode[0] - qmode[1]) > 3:
        qmode[0] = qmode[1]
    if (qmode[N_ang - 1] - qmode[N_ang - 2]) > 3:
        qmode[N_ang - 1] = qmode[N_ang - 2]

    qmode[qmode == 2] = 0.01

    return qmode


def isofreq_Rp(ctes, params_opt):
    """
    Calculate isochromatic reflection coefficient modes.
    """
    q_mode_1 = isofreq_Fresnel_coeff(ctes)

    return q_mode_1


def isofreq_Fresnel_coeff(ctes):
    """
    Calculate Fresnel coefficients on isochromatic contour.
    Scans over angles and radial wave vector magnitudes.
    """
    Mr = 200  # Number of radial wave vector points (adjustable)
    qr0max = 50  # Maximum radial wave vector magnitude (adjustable)

    qr0 = np.linspace(0, qr0max, Mr)

    N_ang = 1000
    al0 = np.linspace(0, 2 * np.pi, N_ang)
    al1 = al0[range(500)]

    q_vals = []
    er_all = []

    for ang in al1:
        Eqiz, shiz, Eiz = Fresnel_coeff_ctes(ang, ctes)

        error = np.zeros(Mr, dtype=complex)

        for jr in range(Mr):
            q_real0 = qr0[jr]
            error[jr] = Fresnel_coeff(Eqiz, shiz, Eiz, q_real0)

        # Ignore NaN values
        valid_mask = ~np.isnan(error)
        Pr = int(np.nanargmax(np.abs(error[valid_mask])))
        q = qr0[Pr]
        error = np.imag(error)

        q_vals.append(q)
        er_all.append(error)

    er_all = np.array(er_all)

    q_mode_1 = RP_first_mode(qr0, al1, er_all)

    return q_mode_1


# =============================================================================
# MINUIT
# =============================================================================


def refine_gd_minuit(shiz, C, q_real0, q_imag0):
    """
    Refine wave vector using Minuit minimization algorithm.
    Minimizes absolute reflection coefficient to find resonance points.
    """
    # Create partial functions with fixed layer constants
    R_colinear = partial(R_trilayer, shiz, C)
    R_abs_colinear = partial(R_abs_trilayer, shiz, C)

    m = Minuit(R_abs_colinear, q_real=q_real0, q_imag=q_imag0)

    m.limits["q_real"] = (0, None)
    m.limits["q_imag"] = (0, None)

    m.migrad()
    q_real = m.values["q_real"]
    q_imag = m.values["q_imag"]

    R_min = R_colinear(q_real, q_imag)

    return q_real, q_imag, R_min


def find_argumument_min_qs(qs):
    """
    Find index of minimum wave vector magnitude.
    Penalizes zero-magnitude wave vectors.
    """
    to_add = np.zeros(qs.shape[0])
    for ind in range(qs.shape[0]):
        # Penalize zero magnitude wave vectors
        if qs[ind, 0] ** 2 + qs[ind, 1] ** 2 == 0:
            to_add[ind] = 1000
    qs_mod = (
        np.array([qs[i, 0] ** 2 + qs[i, 1] ** 2 for i in range(qs.shape[0])]) + to_add
    )
    arg_min_qs = np.argmin(qs_mod)
    return arg_min_qs


# =============================================================================
# Quality factors
# =============================================================================
# Bound the magnitude of Q real and imaginary components.
# This section computes both bounded and unbounded values.


def normalize_angle(angle):
    """
    Normalize angle(s) to [-π/2, 3π/2) range.
    Works with both scalar and array inputs.
    """
    # Convert to numpy array for unified handling
    arr = np.asarray(angle)
    # Normalize to [0, 2π) range
    arr = np.mod(arr, 2 * np.pi)
    # Shift to [-π/2, 3π/2) range
    arr[arr >= 3 * np.pi / 2] -= 2 * np.pi
    arr[arr < -np.pi / 2] += 2 * np.pi
    # Return scalar if input was scalar
    if np.isscalar(angle):
        return arr.item()
    return arr


def is_even(number):
    """Check if a number is even."""
    return number % 2 == 0


def isofreq_minuit_Rp(ctes, q_root, params_opt):
    """
    Calculate isochromatic reflection modes using Minuit optimization.
    Refines initial wave vector guesses over all angles with quality constraints.
    """
    N = params_opt[0]
    q_real_min = params_opt[1]
    q_real_max = params_opt[2]
    q_imag_max = params_opt[3]

    qs_real = []
    qs_imag = []

    # Use q_root as initial guesses for each angle
    al = np.linspace(0, 2 * np.pi, N)

    Cont = 0

    for i in range(500):  # Half of total angles (N/2 due to symmetry)
        shiz, C = C_shiz_trilayer(ctes, al[i])
        q_real0 = q_root[i]

        if i > 0 and q_root[i] < 0.01:
            q_real0 = qs_real[i - 1]

        # Use previous imaginary part if stable, otherwise start with default
        if (i > 1) and (qs_real[i - 1] > 0.1):
            q_imag0 = qs_imag[i - 1]
        else:
            q_imag0 = 0.5

        q_real, q_imag, R_min = refine_gd_minuit(shiz, C, q_real0, q_imag0)

        # Apply quality constraints on the refined values
        if (
            (q_real > q_real_max)
            or (q_imag > q_imag_max)
            or (q_real < q_real_min)
            or (q_real0 == 0)
            or (np.abs(q_real0 - q_real) > 5)  # Avoid large jumps
        ):
            # Reset to small values if constraints violated
            q_real = 0.0001
            q_imag = 0.0001

        qs_real.append(q_real)
        qs_imag.append(q_imag)

        if q_root[i] < 0.01 and q_real == 0.0001 and not Rp:
            Cont = Cont + 1

        if Cont > 0 and (q_real > 0.0001):
            for j in range(Cont):
                shiz, C = C_shiz_trilayer(ctes, al[i - j - 1])
                q_real0 = q_real
                q_imag0 = q_imag
                q_real, q_imag, R_min = refine_gd_minuit(shiz, C, q_real0, q_imag0)

                if (
                    (q_real > q_real_max)
                    or (q_imag > q_imag_max)
                    or (q_real < q_real_min)
                    or (q_real0 == 0)
                    or (np.abs(q_real0 - q_real) > 5)  # Avoid large jumps
                ):
                    Cont = 0
                    break

                qs_real[i - j - 1] = q_real
                qs_imag[i - j - 1] = q_imag

    # Extend arrays to full 2π range (mirror symmetry)
    qs_real.extend(qs_real)
    qs_imag.extend(qs_imag)

    qs_real = np.array(qs_real)
    qs_imag = np.array(qs_imag)

    return qs_real, qs_imag
