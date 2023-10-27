import numpy as np
from math import pi


class BiotSavartQuasiSymmetricFieldDifference():

    def __init__(self, quasi_symmetric_field, biotsavart):
        self.quasi_symmetric_field = quasi_symmetric_field
        self.biotsavart = biotsavart

    def J_L2(self):
        Bbs        = self.biotsavart.B
        Bqs        = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        return np.sum(arc_length[:, None] * (Bbs-Bqs)**2)/len(arc_length)

    def dJ_L2_by_dcoilcoefficients(self):
        Bbs                = self.biotsavart.B
        Bqs                = self.quasi_symmetric_field.B
        dBbs_by_dcoilcoeff = self.biotsavart.dB_by_dcoilcoeffs
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (Bbs-Bqs) * arc_length[:, None]
        for dB in dBbs_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj->k', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dcoilcurrents(self):
        Bbs                   = self.biotsavart.B
        dBbs_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        Bqs                   = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (Bbs-Bqs) * arc_length[:, None]
        for dB in dBbs_by_dcoilcurrents:
            res.append(np.einsum('ij,ij', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dmagneticaxiscoefficients(self):
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        Bbs        = self.biotsavart.B
        dBbs_by_dX = self.biotsavart.dB_by_dX
        Bqs        = self.quasi_symmetric_field.B
        dBqs_by_dcoeffs = self.quasi_symmetric_field.dB_by_dcoeffs

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = 2*np.einsum('ij,ikj,imk,i->m', (Bbs-Bqs), dBbs_by_dX, dgamma_by_dcoeff, arc_length)
        res -= 2*np.einsum('ij,imj,i->m', (Bbs-Bqs), dBqs_by_dcoeffs, arc_length)
        res += np.einsum('i,i,imj,ij->m', (1/arc_length), np.sum((Bbs-Bqs)**2, axis=1), d2gamma_by_dphidcoeff, dgamma_by_dphi)
        res *= 1/arc_length.shape[0]
        return res

    def dJ_L2_by_detabar(self):
        Bbs             = self.biotsavart.B
        arc_length      = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        Bqs             = self.quasi_symmetric_field.B
        dBqs_by_detabar = self.quasi_symmetric_field.dB_by_detabar
        res = np.zeros((1, ))
        res[0] -= np.sum(2*(Bbs-Bqs)*dBqs_by_detabar[:, 0, :] * arc_length[:, None])
        res *= 1/arc_length.shape[0]
        return res

    def J_H1(self):
        dBbs_by_dX = self.biotsavart.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        dBqs_by_dX = self.quasi_symmetric_field.dB_by_dX
        return np.sum(arc_length[:, None, None] * (dBbs_by_dX-dBqs_by_dX)**2)/len(arc_length)

    def dJ_H1_by_dcoilcoefficients(self):
        dBbs_by_dX            = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcoeff = self.biotsavart.d2B_by_dXdcoilcoeffs
        dBqs_by_dX            = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (arc_length[:, None, None])*(dBbs_by_dX-dBqs_by_dX)
        for dB in d2Bbs_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk->l', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dcoilcurrents(self):
        dBbs_by_dX               = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents
        dBqs_by_dX               = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in d2Bbs_by_dXdcoilcurrents:
            res.append(np.einsum('ijk,ijk,i', dBbs_by_dX-dBqs_by_dX, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dmagneticaxiscoefficients(self):

        gamma                 = self.quasi_symmetric_field.magnetic_axis.gamma
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        d2Bqs_by_dcoeffsdX    = self.quasi_symmetric_field.d2B_by_dcoeffsdX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        dBbs_by_dX    = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        dBqs_by_dX    = self.quasi_symmetric_field.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = 2 * np.einsum('ijk,ijlk,iml,i->m',(dBbs_by_dX-dBqs_by_dX), d2Bbs_by_dXdX, dgamma_by_dcoeff, arc_length)
        res -= 2*np.einsum('ijk,imjk,i->m', (dBbs_by_dX-dBqs_by_dX), d2Bqs_by_dcoeffsdX, arc_length)
        res += np.einsum('i,i,iml,il->m', (1/arc_length), np.sum(np.sum((dBbs_by_dX-dBqs_by_dX)**2, axis=1), axis=1), d2gamma_by_dphidcoeff, dgamma_by_dphi)
        res *= 1/gamma.shape[0]
        return res

    def dJ_H1_by_detabar(self):
        dBbs_by_dX         = self.biotsavart.dB_by_dX
        dBqs_by_dX         = self.quasi_symmetric_field.dB_by_dX
        d2Bqs_by_detabardX = self.quasi_symmetric_field.d2B_by_detabardX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        res = np.zeros((1, ))
        res[0] -= np.sum(2*(dBbs_by_dX-dBqs_by_dX)*d2Bqs_by_detabardX[:, 0, :, :] * arc_length[:, None, None])
        res *= 1/arc_length.shape[0]
        return res


class SquaredMagneticFieldNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        B = self.biotsavart.compute(quadrature_points).B
        return np.sum(arc_length[:, None] * (B**2))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        B = self.biotsavart.compute(quadrature_points).B
        dB_by_dcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).dB_by_dcoilcoeffs
        res = []
        for dB in dB_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj,i->k', B, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        B        = self.biotsavart.B
        dB_by_dX = self.biotsavart.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    res[i] += 2 * np.sum(B[:, k1] * dB_by_dX[:, k1, k2] * dgamma_by_dcoeff[:, i, k2] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(B**2, axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res

class SquaredMagneticFieldGradientNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |∇B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        return np.sum(arc_length * (np.sum(np.sum(dB_by_dX**2, axis=1), axis=1)))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        d2B_by_dXdcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).d2B_by_dXdcoilcoeffs
        res = []
        for dB in d2B_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk,i->l', dB_by_dX, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the gradient of the
        magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        dB_by_dX = self.biotsavart.dB_by_dX
        d2B_by_dXdX = self.biotsavart.d2B_by_dXdX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    for k3 in range(3):
                        res[i] += 2.0 * np.sum(dB_by_dX[:, k1, k2] * d2B_by_dXdX[:, k1, k2, k3] * dgamma_by_dcoeff[:, i, k3] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(np.sum(dB_by_dX**2, axis=1), axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res


class CurveLength():

    r"""
    J = \int_{curve} 1 ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        return np.mean(arc_length)

    def dJ_by_dcoefficients(self):
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        for i in range(num_coeff):
            res[i] = np.mean((1/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        return res


class CurveCurvature():

    r"""
    J = \int_{curve} max(\kappa-desired_kappa, 0)^2 ds
    """

    def __init__(self, curve, desired_kappa=None, p=2, root=False):
        self.curve = curve
        self.desired_kappa = desired_kappa
        self.p = p
        self.root = root

    def J(self):
        p = self.p
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        kappa = self.curve.kappa[:, 0]
        if self.root:
            return np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)**(1./p)
        else:
            return np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)

    def dJ_by_dcoefficients(self):
        p = self.p
        kappa                 = self.curve.kappa[:,0]
        dkappa_by_dcoeff      = self.curve.dkappa_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((np.maximum(kappa-self.desired_kappa, 0)**p/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(p*(np.maximum(kappa-self.desired_kappa, 0))**(p-1) * dkappa_by_dcoeff[:,i] * arc_length)
            if self.root:
                res[i] *= (1./p) * np.mean(np.maximum(kappa-self.desired_kappa, 0)**p * arc_length)**(1./p-1)
        return res


class MeanSquaredCurveCurvature():

    r"""
    J = \int_{curve} \kappa ds
    """

    def __init__(self, curve, p=2, root=False):
        self.curve = curve
        self.root = root

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        kappa = self.curve.kappa[:, 0]
        return np.mean(kappa**2 * arc_length) / np.mean(arc_length)

    def dJ_by_dcoefficients(self):
        kappa                 = self.curve.kappa[:,0]
        dkappa_by_dcoeff      = self.curve.dkappa_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)
        
        num = np.mean(kappa**2 * arc_length)
        denom = np.mean(arc_length)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res_num       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res_num[i]  = np.mean((kappa**2/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res_num[i] += np.mean(2*kappa * dkappa_by_dcoeff[:,i] * arc_length)

        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res_denom = np.zeros((num_coeff, ))
        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        for i in range(num_coeff):
            res_denom[i] = np.mean((1/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        return (res_num * denom - res_denom * num) / denom**2




class CurveTorsion():

    r"""
    J = \int_{curve} \tau^p ds
    """

    def __init__(self, curve, p=2, root=False):
        self.curve = curve
        self.p = p
        self.root = root

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        torsion    = self.curve.torsion[:, 0]
        if self.root:
            return np.mean(np.abs(torsion)**self.p * arc_length)**(1./self.p)
        else:
            return np.mean(np.abs(torsion)**self.p * arc_length)

    def dJ_by_dcoefficients(self):
        torsion               = self.curve.torsion[:,0]
        dtorsion_by_dcoeff    = self.curve.dtorsion_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((np.abs(torsion)**self.p/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(self.p*np.abs(torsion)**(self.p-1) * np.sign(torsion) * dtorsion_by_dcoeff[:,i] * arc_length)
            if self.root:
                    res[i] *= (1./self.p) * np.mean(np.abs(torsion)**self.p * arc_length)**(1./self.p - 1.)
        return res


class SobolevTikhonov():

    def __init__(self, curve, weights=[1., 1., 0., 0.]):
        self.curve = curve
        if not len(weights) == 4:
            raise ValueError(
                "You should pass 4 weights: for the L^2, H^1, H^2 and H^3 norm.")
        self.weights = weights
        self.initial_curve = (curve.gamma.copy(), curve.dgamma_by_dphi.copy(
        ), curve.d2gamma_by_dphidphi.copy(), curve.d3gamma_by_dphidphidphi)

    def J(self):
        res = 0
        curve = self.curve
        num_points = curve.gamma.shape[0]
        weights = self.weights
        if weights[0] > 0:
            res += weights[0] * \
                np.sum((curve.gamma-self.initial_curve[0])**2)/num_points
        if weights[1] > 0:
            res += weights[1] * np.sum((curve.dgamma_by_dphi -
                                        self.initial_curve[1])**2)/num_points
        if weights[2] > 0:
            res += weights[2] * np.sum((curve.d2gamma_by_dphidphi -
                                        self.initial_curve[2])**2)/num_points
        if weights[3] > 0:
            res += weights[3] * np.sum((curve.d3gamma_by_dphidphidphi -
                                        self.initial_curve[3])**2)/num_points
        return res

    def dJ_by_dcoefficients(self):
        curve = self.curve
        num_coeff = curve.dgamma_by_dcoeff.shape[1]
        num_points = curve.gamma.shape[0]
        res = np.zeros((num_coeff, ))
        weights = self.weights
        if weights[0] > 0:
            for i in range(num_coeff):
                res[i] += weights[0] * np.sum(
                    2*(curve.gamma-self.initial_curve[0])*curve.dgamma_by_dcoeff[:, i, :])/num_points
        if weights[1] > 0:
            for i in range(num_coeff):
                res[i] += weights[1] * np.sum(2*(curve.dgamma_by_dphi-self.initial_curve[1])
                                              * curve.d2gamma_by_dphidcoeff[:, :, i, :])/num_points
        if weights[2] > 0:
            for i in range(num_coeff):
                res[i] += weights[2] * np.sum(2*(curve.d2gamma_by_dphidphi-self.initial_curve[2])
                                              * curve.d3gamma_by_dphidphidcoeff[:, :, :, i, :])/num_points
        if weights[3] > 0:
            for i in range(num_coeff):
                res[i] += weights[3] * np.sum(2*(curve.d3gamma_by_dphidphidphi-self.initial_curve[3])
                                              * curve.d4gamma_by_dphidphidphidcoeff[:, :, :, :, i, :])/num_points
        return res


class UniformArclength():

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        num_points = self.curve.gamma.shape[0]
        mean_arc = np.mean(self.curve.incremental_arclength)
        return np.sum((self.curve.incremental_arclength-mean_arc)**2)/num_points

    def dJ_by_dcoefficients(self):
        num_points = self.curve.gamma.shape[0]
        num_coeff = self.curve.dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        mean_arc = np.mean(self.curve.incremental_arclength)
        for i in range(num_coeff):
            mean_arc_dcoeff = np.mean(self.curve.dincremental_arclength_by_dcoeff[:, i, :], axis=0)
            res[i] = np.sum(
                2 * (self.curve.incremental_arclength-mean_arc)
                * (self.curve.dincremental_arclength_by_dcoeff[:, i, :] - mean_arc_dcoeff)
            )/num_points
        return res




#class MinimumDistance_ma():
#
#    def __init__(self, curves, ma,  minimum_distance_c, minimum_distance_a):
#        self.curves = curves + [ma]
#        self.minimum_distance_c = minimum_distance_c
#        self.minimum_distance_a = minimum_distance_a
#
#    def min_dist_axis(self):
#        res = 1e10
#
#        gamma1 = self.curves[-1].gamma
#        for j in range(len(self.curves)-1):
#            gamma2 = self.curves[j].gamma
#            dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
#            res = min(res, np.min(dists))
#        return res
#
#    def min_dist_coils(self):
#        res = 1e10
#        for i in range(len(self.curves)-1):
#            gamma1 = self.curves[i].gamma
#            for j in range(i):
#                gamma2 = self.curves[j].gamma
#                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
#                res = min(res, np.min(dists))
#        return res
#
#    def J(self):
#        from scipy.spatial.distance import cdist
#        res = 0
#        for i in range(len(self.curves)):
#            gamma1 = self.curves[i].gamma
#            min_dist = self.minimum_distance_c if i < len(self.curves)-1 else self.minimum_distance_a
#
#            for j in range(i):
#                gamma2 = self.curves[j].gamma
#                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
#                res += np.sum(np.maximum(min_dist-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])
#        return res
#
#    def dJ_by_dcoefficients(self):
#        res = []
#        for i in range(len(self.curves)):
#            gamma1 = self.curves[i].gamma
#            dgamma1 = self.curves[i].dgamma_by_dcoeff
#            numcoeff1 = self.curves[i].dgamma_by_dcoeff.shape[1]
#            res.append(np.zeros((numcoeff1, )))
#            min_dist = self.minimum_distance_c if i < len(self.curves)-1 else self.minimum_distance_a
#            
#            for j in range(i):
#                gamma2 = self.curves[j].gamma
#                dgamma2 = self.curves[j].dgamma_by_dcoeff
#                numcoeff2 = self.curves[j].dgamma_by_dcoeff.shape[1]
#                diffs = gamma1[:, None, :] - gamma2[None, :, :]
#
#                dists = np.sqrt(np.sum(diffs**2, axis=2))
#                if np.sum(np.maximum(min_dist - dists, 0)) < 1e-15:
#                    continue
#
#                for ii in range(numcoeff1):
#                    res[i][ii] += np.sum(-2 * np.maximum(min_dist - dists, 0) * np.sum(dgamma1[:, ii, :][:, None, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
#                for jj in range(numcoeff2):
#                    res[j][jj] -= np.sum(-2 * np.maximum(min_dist - dists, 0) * np.sum(dgamma2[:, jj, :][None, :, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
#
#        res_coils = res[:-1]
#        res_ma = np.array(res[-1])
#        return res_coils, res_ma









class MinimumDistance_ma():

    def __init__(self, curves, ma,  minimum_distance):
        self.curves = curves + [ma]
        self.minimum_distance = minimum_distance

    def min_dist(self):
        res = 1e10
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res

    def J(self):
        from scipy.spatial.distance import cdist
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res += np.sum(np.maximum(self.minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])
        return res

    def dJ_by_dcoefficients(self):
        res = []
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            dgamma1 = self.curves[i].dgamma_by_dcoeff
            numcoeff1 = self.curves[i].dgamma_by_dcoeff.shape[1]
            res.append(np.zeros((numcoeff1, )))
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dgamma2 = self.curves[j].dgamma_by_dcoeff
                numcoeff2 = self.curves[j].dgamma_by_dcoeff.shape[1]
                diffs = gamma1[:, None, :] - gamma2[None, :, :]

                dists = np.sqrt(np.sum(diffs**2, axis=2))
                if np.sum(np.maximum(self.minimum_distance - dists, 0)) < 1e-15:
                    continue

                for ii in range(numcoeff1):
                    res[i][ii] += np.sum(-2 * np.maximum(self.minimum_distance - dists, 0) * np.sum(dgamma1[:, ii, :][:, None, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
                for jj in range(numcoeff2):
                    res[j][jj] -= np.sum(-2 * np.maximum(self.minimum_distance - dists, 0) * np.sum(dgamma2[:, jj, :][None, :, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])

        res_coils = res[:-1]
        res_ma = np.array(res[-1])
        return res_coils, res_ma

    def min_dist_axis(self):
        res = 1e10

        gamma1 = self.curves[-1].gamma
        for j in range(len(self.curves)-1):
            gamma2 = self.curves[j].gamma
            dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
            res = min(res, np.min(dists))
        return res

    def min_dist_coils(self):
        res = 1e10
        for i in range(len(self.curves)-1):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res



class MinimumDistance():

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.minimum_distance = minimum_distance

    def min_dist(self):
        res = 1e10
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res

    def J(self):
        from scipy.spatial.distance import cdist
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res += np.sum(np.maximum(self.minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])
        return res

    def dJ_by_dcoefficients(self):
        res = []
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            dgamma1 = self.curves[i].dgamma_by_dcoeff
            numcoeff1 = self.curves[i].dgamma_by_dcoeff.shape[1]
            res.append(np.zeros((numcoeff1, )))
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dgamma2 = self.curves[j].dgamma_by_dcoeff
                numcoeff2 = self.curves[j].dgamma_by_dcoeff.shape[1]
                diffs = gamma1[:, None, :] - gamma2[None, :, :]

                dists = np.sqrt(np.sum(diffs**2, axis=2))
                if np.sum(np.maximum(self.minimum_distance - dists, 0)) < 1e-15:
                    continue

                for ii in range(numcoeff1):
                    res[i][ii] += np.sum(-2 * np.maximum(self.minimum_distance - dists, 0) * np.sum(dgamma1[:, ii, :][:, None, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
                for jj in range(numcoeff2):
                    res[j][jj] -= np.sum(-2 * np.maximum(self.minimum_distance - dists, 0) * np.sum(dgamma2[:, jj, :][None, :, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
        return res

class CoilLpReduction():

    def __init__(self, objectives, p=2, root=False):
        self.objectives = objectives
        self.p = p
        self.root = root

    def J(self):
        p = self.p
        if self.root:
            return sum([J.J()**p for J in self.objectives])**(1./p)
        else:
            return sum([J.J()**p for J in self.objectives])

    def dJ_by_dcoefficients(self):
        p = self.p
        if self.root:
            return (1./p)*sum([J.J()**p for J in self.objectives])**(1./p-1) * np.concatenate([p*(J.J()**(p-1))*J.dJ_by_dcoefficients() for J in self.objectives], axis=0)
        else:
            return np.concatenate([p*(J.J()**(p-1))*J.dJ_by_dcoefficients() for J in self.objectives], axis=0)
