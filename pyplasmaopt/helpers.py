import os
import numpy as np
from .curve import CartesianFourierCurve, StelleratorSymmetricCylindricalFourierCurve
import cppplasmaopt as cpp
from qsc import Qsc

def get_flat_data(Nt=16, Nt_ma=16, nfp=1, ppp=10, ncoils_per_hp=3, coil_minor_radius=0.5, coil_major_radius=1., axis_radius=1.):
    coils = [CartesianFourierCurve(Nt, np.linspace(0, 1, Nt*ppp, endpoint=False)) for i in range(ncoils_per_hp)]
    
    #flat coils
    hp_angle = 2*np.pi / nfp / 2.
    for ic in range(ncoils_per_hp):
        angle = ((ic+0.5)/ncoils_per_hp) * hp_angle
        coils[ic].coefficients[0][0] = coil_major_radius * np.cos(angle)
        coils[ic].coefficients[1][0] = coil_major_radius * np.sin(angle)
        coils[ic].coefficients[2][0] = 0.
        coils[ic].coefficients[0][2] = coil_minor_radius *  np.cos(angle)
        coils[ic].coefficients[1][2] = coil_minor_radius *  np.sin(angle)
        coils[ic].coefficients[2][1] = coil_minor_radius
        coils[ic].update()
    
    # flat axis
    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False))
    ma.coefficients[0][0] = axis_radius
    ma.coefficients[1][0] = 0.
    ma.update()
    
    # same as ma, but with quadrature points on the full domain [0, 2\pi)
    ma_full_torus = StelleratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1., nfp*numpoints, endpoint=False))
    ma_full_torus.coefficients[0][0] = axis_radius
    ma_full_torus.coefficients[1][0] = 0.
    ma_full_torus.update()

    currents = np.zeros((ncoils_per_hp,))
    return (coils, ma, ma_full_torus, currents)

def linking_number(coils):
    try:
        ln_coils = 0
        for i in range(len(coils)):
            for j in range(i+1, len(coils)):
                ln_coils += abs(cpp.ln(coils[i].gamma, coils[j].gamma))
    except:
        ln_coils=100
    return ln_coils

def get_helicity(axis, etabar):
    try:
        order = axis.order
        dofs = axis.get_dofs()
        rc = dofs[:order+1]
        zs = dofs[order+1:]
        qsc = Qsc(rc, np.insert(zs, 0, 0), nfp=axis.nfp, etabar=etabar)
        hel = abs(int(qsc.helicity))
    except:
        hel=100
    return hel


def get_simsopt_coils(base_curves, base_currents, nfp, stellsym, num_points=None):
    from simsopt.geo import CurveXYZFourier, CurveRZFourier
    from simsopt.field.coil import Current, ScaledCurrent, coils_via_symmetries
    
    ncoils = len(base_curves)
    matlabcoils = [c.tomatlabformat() for c in base_curves]
    coil_data = np.concatenate(matlabcoils, axis=-1)
    scoils = [CurveXYZFourier(c.gamma.shape[0], c.order) if num_points is None else CurveXYZFourier(num_points, c.order)  for c in base_curves]
    for ic in range(ncoils):
        dofs = scoils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, coil_data.shape[0]-1):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        scoils[ic].local_x = np.concatenate(dofs)
    
    scurrents = [ScaledCurrent(Current(c), 1./(4 * np.pi * 1e-7)) for c in base_currents]
    coils = coils_via_symmetries(scoils, scurrents, nfp, stellsym)
    return coils

def get_simsopt_curveRZ(curve):
    from simsopt.geo import CurveRZFourier
    axis_curve = CurveRZFourier(curve.gamma.shape[0], curve.order, curve.nfp, True)
    axis_curve.rc = curve.coefficients[0]
    axis_curve.zs = curve.coefficients[1]
    axis_curve.local_full_x = axis_curve.get_dofs()
    return axis_curve
