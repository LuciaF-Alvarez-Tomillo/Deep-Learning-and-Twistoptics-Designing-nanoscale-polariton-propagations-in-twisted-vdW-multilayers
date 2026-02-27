# -*- coding: utf-8 -*-
"""
===========================================================
Permitivities
===========================================================
This script contains all the permittivity functions and their
parameters used in the document.

Contains functions for the permittivity of:
    SiO2
    Au
    CoTeMoO6
    V2O5
    MoO3 in IR and THz regimes
based on experimental data and oscillator models.
Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# =============================================================================
# ########## Permittivity Functions ##########
# =============================================================================


def _default_data_dir() -> Path:
    """Return repository data directory used for permittivity tables."""
    return Path(__file__).resolve().parent[2] / "data"


# Lorentz oscillator model
def epsilon_1phonon(w, wT, wL, gT, gL):
    eps = (w**2 - wL**2 + 1j * gL * w) / (w**2 - wT**2 + 1j * gT * w)
    return eps


# =============================================================================
# SiO2
# =============================================================================
def eps_SiO2_exp(w):
    """
    Permittivity of SiO2 (experimental).
    Data source: Nanomaterials 2021, 11(1), 120. https://doi.org/10.3390/nano11010120
    """
    eps = 1.0
    eps_inf = 2
    gamma = [51, 10, 10]
    wTO = [450, 800, 1045]
    wLO = [505, 830, 1240]

    eps = eps_inf
    for i in range(0, len(wTO)):
        eps = eps * epsilon_1phonon(w, wTO[i], wLO[i], gamma[i], gamma[i])
    return eps


# =============================================================================
# Au
# =============================================================================
def eps_Au(w):
    """
    Permittivity of Au (gold).
    Data source: Babar, S., & Weaver, J. H. (2015).
    Optical constants of Cu, Ag, and Au revisited.
    Applied Optics, 54(3), 477-481. https://doi.org/10.1364/AO.54.000477
    Data file from: https://refractiveindex.info/
    """
    data = np.loadtxt(_default_data_dir() / "epsAu.txt")

    # Colums
    frecuencia = data[:, 0]
    real_part = data[:, 1]
    imag_part = data[:, 2]

    # Interpolate
    re_interp = interp1d(frecuencia, real_part, kind="cubic", fill_value="extrapolate")
    im_interp = interp1d(frecuencia, imag_part, kind="cubic", fill_value="extrapolate")

    eps = re_interp(w) + 1j * im_interp(w)

    return eps


# =============================================================================
# # CoTeMoO6
# =============================================================================


def epsilon_Co(w, wi, fi, gi):
    eps = fi**2 / (wi**2 - w**2 - 1j * gi * w)
    return eps


def eps_CoTeMoO6(w, Axis):
    """
    Permittivity of CoTeMoO6 for different axes.
    Data source: Wang, Y. et al. (2024).
    Giant in-plane optical anisotropy in a van der Waals antiferromagnet.
    Nature Nanotechnology. https://doi.org/10.1038/s41565-024-01628-y
    """
    if Axis == "A":
        eps = eps_X_CoTeMoO6(w)
    if Axis == "B":
        eps = eps_Y_CoTeMoO6(w)
    if Axis == "C":
        eps = eps_Z_CoTeMoO6(w)
    return eps


def eps_X_CoTeMoO6(w):
    """
    Permittivity of CoTeMoO6 along X axis.
    Data source: Wang, Y. et al. (2024). Nature Nanotechnology. https://doi.org/10.1038/s41565-024-01628-y
    """
    epsAinf = 3.3
    eps = 0.0
    wix = [676.47, 712.74, 891.90]
    fix = [582.04, 140.30, 493.63]
    gix = [4.09, 22.71, 3.21]
    for i in range(0, len(wix)):
        eps = eps + epsilon_Co(w, wix[i], fix[i], gix[i])
    eps = epsAinf + eps
    return eps


def eps_Y_CoTeMoO6(w):
    """
    Permittivity of CoTeMoO6 along Y axis.
    Data source: Wang, Y. et al. (2024). Nature Nanotechnology. https://doi.org/10.1038/s41565-024-01628-y
    """
    epsBinf = 4.95
    eps = 0.0
    wiy = [650.54, 712.08, 897.98]
    fiy = [938.71, 159.53, 382.50]
    giy = [2.11, 28.80, 2.79]
    for i in range(0, len(wiy)):
        eps = eps + epsilon_Co(w, wiy[i], fiy[i], giy[i])
    eps = epsBinf + eps
    return eps


def eps_Z_CoTeMoO6(w):
    """
    Permittivity of CoTeMoO6 along Z axis.
    Data source: Wang, Y. et al. (2024). Nature Nanotechnology. https://doi.org/10.1038/s41565-024-01628-y
    """
    epsCinf = 2.4
    eps = 0.0
    wiz = [938]
    fiz = [252]
    giz = [8]
    for i in range(0, len(wiz)):
        eps = eps + epsilon_Co(w, wiz[i], fiz[i], giz[i])
    eps = epsCinf + eps
    return eps


# =============================================================================
# V2O5
# =============================================================================
def eps_A_V2O5_4parOsc(w):
    """
    Permittivity of V2O5 along A axis (4-parameter oscillator model).
    Data source: Clauws, P., & Vennik, J. (1974). Lattice Vibrations of V2O5.
    Determination of TO and LO Frequencies from Infrared Reflection and Transmission.
    physica status solidi (b), 65(2), 677-686.  https://doi.org/10.1002/pssb.2220760232
    """
    epsAinf = 5.5
    eps = 1.0
    GTOA = [3.6, 13, 15, 5, 30, 10]
    GLOA = [4.2, 8, 12.2, 30, 50, 15]
    wTOA = [72.4, 261, 303, 411, 767.5, 980.5]
    wLOA = [76.2, 265.5, 390.5, 586, 959, 982]
    for i in range(0, len(wTOA)):
        eps = eps * epsilon_1phonon(w, wTOA[i], wLOA[i], GTOA[i], GLOA[i])
    eps = epsAinf * eps
    return eps


def eps_B_V2O5_4parOsc(w):
    """
    Permittivity of V2O5 along B axis (4-parameter oscillator model).
    Data source: Clauws, P., & Vennik, J. (1974). physica status solidi (b), 65(2), 677-686.  https://doi.org/10.1002/pssb.2220760232
    """
    epsBinf = 4.77
    eps = 1.0
    GTOB = [18, 2.5]
    GLOB = [15, 2.5]
    wTOB = [473, 975.5]
    # wTOB = [473,  960];
    wLOB = [490, 1038]
    for i in range(0, len(wTOB)):
        eps = eps * epsilon_1phonon(w, wTOB[i], wLOB[i], GTOB[i], GLOB[i])
    eps = epsBinf * eps
    return eps


def eps_C_V2O5_4parOsc(w):
    """
    Permittivity of V2O5 along C axis (4-parameter oscillator model).
    Data source: Clauws, P., & Vennik, J. (1974). physica status solidi (b), 65(2), 677-686.  https://doi.org/10.1002/pssb.2220760232
    """
    epsCinf = 4.49
    eps = 1.0
    GTOC = [10.5, 7.8, 21.0]
    GLOC = [7.5, 10.2, 18.0]
    wTOC = [212.0, 284.0, 506.5]
    wLOC = [225.0, 315.5, 842.5]
    for i in range(0, len(wTOC)):
        eps = eps * epsilon_1phonon(w, wTOC[i], wLOC[i], GTOC[i], GLOC[i])
    eps = epsCinf * eps
    return eps


def eps_V2O5_4parOsc(w, Axis):
    """
    Permittivity of V2O5 (4-parameter oscillator model) for a given axis.
    Data source: Clauws, P., & Vennik, J. (1974). physica status solidi (b), 65(2), 677-686.  https://doi.org/10.1002/pssb.2220760232
    """
    if Axis == "A":
        eps = eps_A_V2O5_4parOsc(w)
    if Axis == "B":
        eps = eps_C_V2O5_4parOsc(w)
    if Axis == "C":
        eps = eps_B_V2O5_4parOsc(w)
    return eps


# =============================================================================
#  MoO3
# =============================================================================


def eps_A_2(w):
    """
    Permittivity of MoO3 along XX (A) axis (4-parameter oscillator model).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsXinf = 5.78
    eps = 1.0
    GTOX = [49.1, 6, 0.35]
    GLOX = GTOX
    wTOX = [506.7, 821.4, 998.7]
    wLOX = [534.3, 963, 999.2]

    for i in range(0, len(wTOX)):
        eps = eps * epsilon_1phonon(w, wTOX[i], wLOX[i], GTOX[i], GLOX[i])

    eps = epsXinf * eps
    return eps


def eps_B_2(w):
    """
    Permittivity of MoO3 along YY (B) axis (4-parameter oscillator model).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsYinf = 6.07
    eps = 1.0
    GTOY = [9.5]
    GLOY = GTOY
    wTOY = [544.6]
    wLOY = [850.1]
    for i in range(0, len(wTOY)):
        eps = eps * epsilon_1phonon(w, wTOY[i], wLOY[i], GTOY[i], GLOY[i])
    eps = epsYinf * eps
    return eps


def eps_C_2(w):
    """
    Permittivity of MoO3 along ZZ (C) axis (4-parameter oscillator model).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsZinf = 4.47
    eps = 1.0
    GTOZ = [1.5]
    GLOZ = GTOZ
    wTOZ = [956.7]
    wLOZ = [1006.9]
    for i in range(0, len(wTOZ)):
        eps = eps * epsilon_1phonon(w, wTOZ[i], wLOZ[i], GTOZ[i], GLOZ[i])
    eps = epsZinf * eps
    return eps


def eps_MoO3_exp_IR_THz_2(w, Axis):  # for RB1 & RB2 regime
    """
    Permittivity of alpha-MoO3 in THz and IR (4-parameter Lorentz oscillator model).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    if Axis == "A":
        eps = eps_A_2(w)
    if Axis == "B":
        eps = eps_B_2(w)
    if Axis == "C":
        eps = eps_C_2(w)
    return eps


# MoO3_THz


def eps_A_3(w):
    """
    Permittivity of MoO3 along XX (A) axis (extended 4-parameter oscillator model for THz regime).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsXinf = 5.78
    eps = 1.0
    GTOX = [3, 49.1, 6, 0.35]
    GLOX = GTOX
    wTOX = [367, 506.7, 821.4, 998.7]
    wLOX = [390, 534.3, 963, 999.2]

    for i in range(0, len(wTOX)):
        eps = eps * epsilon_1phonon(w, wTOX[i], wLOX[i], GTOX[i], GLOX[i])

    eps = epsXinf * eps
    return eps


def eps_B_3(w):
    """
    Permittivity of MoO3 along YY (B) axis (extended 4-parameter oscillator model for THz regime).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsYinf = 6.07
    eps = 1.0
    GTOY = [4, 9.5]
    GLOY = GTOY
    wTOY = [262, 544.6]
    wLOY = [367, 850.1]
    for i in range(0, len(wTOY)):
        eps = eps * epsilon_1phonon(w, wTOY[i], wLOY[i], GTOY[i], GLOY[i])
    eps = epsYinf * eps
    return eps


def eps_C_3(w):
    """
    Permittivity of MoO3 along ZZ (C) axis (extended 4-parameter oscillator model for THz regime).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    epsZinf = 4.47
    eps = 1.0
    GTOZ = [1, 1.5]
    GLOZ = GTOZ
    wTOZ = [337, 956.7]
    wLOZ = [363, 1006.9]
    for i in range(0, len(wTOZ)):
        eps = eps * epsilon_1phonon(w, wTOZ[i], wLOZ[i], GTOZ[i], GLOZ[i])
    eps = epsZinf * eps
    return eps


def eps_MoO3_exp_IR_THz_3(w, Axis):  # for THz regime
    """
    Permittivity of alpha-MoO3 in THz and IR (extended 4-parameter Lorentz oscillator model).
    IR data: Álvarez-Pérez, G. et al. (2020). Advanced Materials, 32(29), 1908176. https://doi.org/10.1002/adma.201908176
    THz data: de Oliveira, J. M. et al. (2020). arXiv:2007.06342
    """
    if Axis == "A":
        eps = eps_A_3(w)
    if Axis == "B":
        eps = eps_B_3(w)
    if Axis == "C":
        eps = eps_C_3(w)
    return eps


# =============================================================================
# # MoOCl2
# =============================================================================
def eps_MoOCl2(w, Axis):
    """
    Permittivity of MoOCl2 for a given axis, using tabulated data files.
    Data source: [Please insert the correct reference for the MoOCl2 permittivity data used.]
    """
    w = 1e7 / w
    # Directory where the data files are located
    data_dir = _default_data_dir() / "MoOCl2_permittivity"

    # Load files
    eps_MoOCl2_xx_real = np.loadtxt(data_dir / "eps_MoOCl2_xx_real.txt")
    eps_MoOCl2_yy_real = np.loadtxt(data_dir / "eps_MoOCl2_yy_real.txt")
    eps_MoOCl2_zz_real = np.loadtxt(data_dir / "eps_MoOCl2_zz_real.txt")
    eps_MoOCl2_xx_imag = np.loadtxt(data_dir / "eps_MoOCl2_xx_imag.txt")
    eps_MoOCl2_yy_imag = np.loadtxt(data_dir / "eps_MoOCl2_yy_imag.txt")
    eps_MoOCl2_zz_imag = np.loadtxt(data_dir / "eps_MoOCl2_zz_imag.txt")

    # Interpolation
    if Axis == "A":
        re_interp = interp1d(
            eps_MoOCl2_xx_real[:, 0], eps_MoOCl2_xx_real[:, 1], kind="cubic"
        )
        im_interp = interp1d(
            eps_MoOCl2_xx_imag[:, 0], eps_MoOCl2_xx_imag[:, 1], kind="cubic"
        )
        eps = re_interp(w) + 1j * im_interp(w)
    elif Axis == "B":
        re_interp = interp1d(
            eps_MoOCl2_yy_real[:, 0], eps_MoOCl2_yy_real[:, 1], kind="cubic"
        )
        im_interp = interp1d(
            eps_MoOCl2_yy_imag[:, 0], eps_MoOCl2_yy_imag[:, 1], kind="cubic"
        )
        eps = re_interp(w) + 1j * im_interp(w)
    elif Axis == "C":
        re_interp = interp1d(
            eps_MoOCl2_zz_real[:, 0], eps_MoOCl2_zz_real[:, 1], kind="cubic"
        )
        im_interp = interp1d(
            eps_MoOCl2_zz_imag[:, 0], eps_MoOCl2_zz_imag[:, 1], kind="cubic"
        )
        eps = re_interp(w) + 1j * im_interp(w)
    else:
        raise ValueError("Axis must be 'A', 'B', or 'C'.")

    return eps
