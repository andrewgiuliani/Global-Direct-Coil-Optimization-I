from .biotsavart import BiotSavart
from .quasi_symmetric_field import QuasiSymmetricField
from .objective import BiotSavartQuasiSymmetricFieldDifference, CurveLength, CurveTorsion, MeanSquaredCurveCurvature, CurveCurvature, SobolevTikhonov, UniformArclength, MinimumDistance, MinimumDistance_ma, CoilLpReduction
from .curve import GaussianSampler
from .stochastic_objective import StochasticQuasiSymmetryObjective, CVaR
from .logging import info

from mpi4py import MPI
from math import pi, sin, cos
import numpy as np
import os
from rich.console import Console
from rich.table import Column, Table

class SimpleNearAxisQuasiSymmetryObjective():

    def __init__(self, stellarator, ma, ma_ft, iota_target, iota_weight=1., eta_bar=-2.25,
                 coil_length_target=None, coil_length_weight=0.,
                 magnetic_axis_radius_target=None, magnetic_axis_radius_weight=0.,
                 target_curvature=None, curvature_weight=0.,
                 msc_target=None, msc_weight=0.,
                 arclength_weight=0.,
                 minimum_distance=None, distance_weight=0.,
                 magnetic_axis_length_target=None, magnetic_axis_length_weight=0.,
                 outdir="output/"
                 ):
        self.stellarator = stellarator
        self.ma = ma       # magnetic axis with quadrature points on a single period [0, 2\pi/nfp)
        self.ma_ft = ma_ft # same as self.ma but with quadrature points on the full [0, 2\pi)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma)
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma = qsf.sigma
        iota = qsf.iota

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        coils = stellarator._base_coils
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_length     = CurveLength(ma)
        self.coil_length_target = coil_length_target
        self.coil_length_targets = [coil_length_target for coil in coils]
        self.magnetic_axis_length_target = magnetic_axis_length_target
        self.magnetic_axis_length_weight = magnetic_axis_length_weight
        self.length_weight = coil_length_weight

        self.J_coil_curvatures = [CurveCurvature(coil, desired_kappa=target_curvature, p=2, root=False) for coil in coils]
        self.J_coil_msc = [MeanSquaredCurveCurvature(coil, p=2, root=False) for coil in coils]

        self.J_arclength = [UniformArclength(coil) for coil in coils]
        self.J_distance = MinimumDistance_ma(stellarator.coils, ma, minimum_distance)

        self.msc_target = msc_target
        
        self.magnetic_axis_radius_weight = magnetic_axis_radius_weight
        self.iota_target                 = iota_target
        self.iota_weight                 = iota_weight
        self.curvature_weight            = curvature_weight
        self.msc_weight                  = msc_weight
        self.num_ma_dofs = len(ma.get_dofs())
        self.current_fak = 1./(4 * pi * 1e-7)

        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellarator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellarator.get_dofs()))

        self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        self.x = self.x0.copy()
        self.arclength_weight = arclength_weight
        self.distance_weight =  distance_weight

        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []
        self.outdir = outdir

    def set_dofs(self, x):
        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_ma[0] = 1.
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        self.t = x[-1]

        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.ma_ft.set_dofs(x_ma)
        self.biotsavart.set_points(self.ma.gamma)
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

        self.biotsavart.clear_cached_properties()
        self.qsf.clear_cached_properties()

    def update(self, x, compute_derivative=True):
        self.x[:] = x
        J_BSvsQS          = self.J_BSvsQS
        J_coil_lengths    = self.J_coil_lengths
        J_axis_length     = self.J_axis_length
        J_coil_curvatures = self.J_coil_curvatures
        J_coil_msc   = self.J_coil_msc

        iota_target                 = self.iota_target
        iota_weight                 = self.iota_weight
        magnetic_axis_length_target = self.magnetic_axis_length_target
        curvature_weight             = self.curvature_weight
        msc_weight               = self.msc_weight
        qsf = self.qsf

        self.set_dofs(x)

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """

        self.res1        = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
        if compute_derivative:
            self.dresetabar  += 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
            self.dresma      += 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
            self.drescoil    += 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) \
                + 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
            self.drescurrent += 0.5 * self.current_fak * (
                self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
            )

        self.res2      = 0.5 * self.length_weight * sum( (1/l)**2 * (J2.J() - l)**2 for (J2, l) in zip(J_coil_lengths, self.coil_length_targets))
        if compute_derivative:
            self.drescoil += self.stellarator.reduce_coefficient_derivatives([
                (1/l)**2 * self.length_weight * (J_coil_lengths[i].J()-l) * J_coil_lengths[i].dJ_by_dcoefficients() for (i, l) in zip(list(range(len(J_coil_lengths))), self.coil_length_targets)])

        self.res3 = 0.5 * self.magnetic_axis_radius_weight * (self.ma.get_dofs()[0] - 1.)**2
        if compute_derivative:
            arr = np.zeros(self.dresma.size)
            arr[0] = 1
            self.dresma += self.magnetic_axis_radius_weight * (self.ma.get_dofs()[0] - 1.) * arr 

        self.res4        = 0.5 * iota_weight * (1/iota_target**2) * (qsf.iota-iota_target)**2
        if compute_derivative:
            self.dresetabar += iota_weight *(1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_detabar[:,0]
            self.dresma     += iota_weight *(1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_dcoeffs[:, 0]

        if curvature_weight > 0:
            self.res5      = sum(curvature_weight * J.J() for J in J_coil_curvatures)
            if compute_derivative:
                self.drescoil += self.curvature_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0

        if msc_weight > 0:
            val = [np.minimum(self.msc_target-J.J(), 0) for J in J_coil_msc]
            self.res6      = sum(0.5*msc_weight * v**2 for (v, J) in zip(val, J_coil_msc))
            if compute_derivative:
                self.drescoil += self.msc_weight * self.stellarator.reduce_coefficient_derivatives([-v * J.dJ_by_dcoefficients() for (v, J) in zip(val, J_coil_msc)])
        else:
            self.res6 = 0

        if self.arclength_weight > 0:
            self.res8 = sum(self.arclength_weight * J.J() for J in self.J_arclength)
            if compute_derivative:
                self.drescoil += self.arclength_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclength])
        else:
            self.res8 = 0

        if self.distance_weight > 0:
            self.res9 = self.distance_weight * self.J_distance.J()
            if compute_derivative:
                res_coils, res_ma = self.J_distance.dJ_by_dcoefficients()
                self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(res_coils)
                self.dresma   += self.distance_weight * res_ma
        else:
            self.res9 = 0
        
        if self.magnetic_axis_length_weight > 0:
            self.res10    = 0.5 *self.magnetic_axis_length_weight* (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
            if compute_derivative:
                self.dresma += self.magnetic_axis_length_weight*(1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()
        else:
            self.res10 = 0.
       
        self.dresma[0] = 0.

        Jvals_individual = [self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res8, self.res9, self.res10]
        self.res = sum(Jvals_individual)

        if compute_derivative:
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil
            ))

    def clear_history(self):
        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []

    def callback(self, x, verbose=True):
        self.update(x)# assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        self.dJvals.append((
            norm(self.dres), norm(self.dresetabar), norm(self.dresma), norm(self.drescurrent), norm(self.drescoil)
        ))
        self.xiterates.append(x.copy())
        
        # save backups for line search
        self.x_backup = x.copy()
        self.J_backup = self.res
        self.dJ_backup = self.dres.copy()
        self.qsf._QuasiSymmetricField__state_backup = self.qsf._QuasiSymmetricField__state.copy()

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.dresma):.6e}, {norm(self.drescurrent):.6e}, {norm(self.drescoil):.6e}")

        res_dict={}
        res_dict['nonQS'] = self.res1
        res_dict['coil length'] = self.res2
        res_dict['ma radius'] = self.res3
        res_dict['iota'] = self.res4
        res_dict['curvature'] = self.res5
        res_dict['msc'] = self.res6
        res_dict['arclength'] = self.res8
        res_dict['min dist'] = self.res9
        res_dict['ma length'] = self.res10
        
        console = Console(width=150)
        table1 = Table(expand=True, show_header=False)
        table1.add_row(*[f"{v}" for v in res_dict.keys()])
        table1.add_row(*[f"{v:.6e}" for v in res_dict.values()])
        console.print(table1)
        
        other_char = {}
        other_char['iota'] = f'{self.qsf.iota:.6e}'
        other_char['eta bar'] = f'{self.qsf.eta_bar:.6e}'
        other_char['total length'] = f'{sum([J.J() for J in self.J_coil_lengths]):.6e}'
        other_char['coil lengths'] = ' '.join([f"{J.J():.6e}" for J in self.J_coil_lengths])
        other_char['minimum distance coils'] = f"{self.J_distance.min_dist_coils():.6e}"
        other_char['minimum distance axis'] = f"{self.J_distance.min_dist_axis():.6e}"
        other_char['curvature'] = " ".join([f"{np.max(c.kappa):.6e}" for c in self.stellarator._base_coils])
        other_char['msc'] = " ".join([f"{msc:.6e}" for msc in [np.mean(c.kappa**2 * np.linalg.norm(c.dgamma_by_dphi, axis=-1))/np.mean(np.linalg.norm(c.dgamma_by_dphi, axis=-1)) for c in self.stellarator._base_coils]])
        other_char['magnetic axis radius'] = f"{self.ma.get_dofs()[0]:.6e}"
        
        table2 = Table(expand=True, show_header=False) 
        for k in other_char.keys():
            table2.add_row(k, other_char[k])
        console.print(table2)

        #if iteration % 3000 == 0:
        #    self.plot('iteration-%04i' % iteration)

    def plot(self, filename, coilpy=True):
        if coilpy:
            coilpy_plot(self.stellarator.coils, self.outdir + filename)
        else:
            curves_to_vtk(self.stellarator.coils, self.outdir + filename, close=False)
        np.savetxt(self.outdir + f"x_{filename}.txt", self.x)
        #self.stellarator.savetotxt(self.outdir)
        matlabcoils = [c.tomatlabformat() for c in self.stellarator._base_coils]
        np.savetxt(os.path.join(self.outdir, f'coilsmatlab_{filename}.txt'), np.hstack(matlabcoils))
        np.savetxt(os.path.join(self.outdir, f'currents_{filename}.txt'), self.stellarator._base_currents)




        def save_to_matlab(self, dirname):
            dirname = os.path.join(self.outdir, dirname)
            os.makedirs(dirname, exist_ok=True)
            matlabcoils = [c.tomatlabformat() for c in self.stellarator._base_coils]
            np.savetxt(os.path.join(dirname, 'coils.txt'), np.hstack(matlabcoils))
            np.savetxt(os.path.join(dirname, 'currents.txt'), self.stellarator._base_currents)
            np.savetxt(os.path.join(dirname, 'eta_bar.txt'), [self.qsf.eta_bar])
            np.savetxt(os.path.join(dirname, 'cR.txt'), self.ma.coefficients[0])
            np.savetxt(os.path.join(dirname, 'sZ.txt'), np.concatenate(([0], self.ma.coefficients[1])))




def coilpy_plot(curves, filename, height=0.05, width=0.05):
    def wrap(data):
        return np.concatenate([data, [data[0]]])
    xx = [wrap(c.gamma[:, 0]) for c in curves]
    yy = [wrap(c.gamma[:, 1]) for c in curves]
    zz = [wrap(c.gamma[:, 2]) for c in curves]
    II = [1. for _ in curves]
    names = [i for i in range(len(curves))]
    from coilpy import Coil
    coils = Coil(xx, yy, zz, II, names, names)
    coils.toVTK(filename+'.vtu', line=False, height=height, width=width)

def curves_to_vtk(curves, filename, close=False):
    """
    Export a list of Curve objects in VTK format, so they can be
    viewed using Paraview. This function requires the python package ``pyevtk``,
    which can be installed using ``pip install pyevtk``.
    Args:
        curves: A python list of Curve objects.
        filename: Name of the file to write.
        close: Whether to draw the segment from the last quadrature point back to the first.
    """
    from pyevtk.hl import polyLinesToVTK

    def wrap(data):
        return np.concatenate([data, [data[0]]])

    if close:
        x = np.concatenate([wrap(c.gamma[:, 0]) for c in curves])
        y = np.concatenate([wrap(c.gamma[:, 1]) for c in curves])
        z = np.concatenate([wrap(c.gamma[:, 2]) for c in curves])
        ppl = np.asarray([c.gamma.shape[0]+1 for c in curves])
    else:
        x = np.concatenate([c.gamma[:, 0] for c in curves])
        y = np.concatenate([c.gamma[:, 1] for c in curves])
        z = np.concatenate([c.gamma[:, 2] for c in curves])
        ppl = np.asarray([c.gamma.shape[0] for c in curves])
    data = np.concatenate([i*np.ones((ppl[i], )) for i in range(len(curves))])
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})



def plot_stellarator(stellarator, axis=None, extra_data=None):
    coils = stellarator.coils
    gamma = coils[0].gamma
    N = gamma.shape[0]
    l = len(stellarator.coils)
    data = np.zeros((l*(N+1), 3))
    labels = [None for i in range(l*(N+1))]
    groups = [None for i in range(l*(N+1))]
    for i in range(l):
        data[(i*(N+1)):((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma
        data[((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma[0, :]
        for j in range(i*(N+1), (i+1)*(N+1)):
            labels[j] = 'Coil %i ' % stellarator.map[i]
            groups[j] = i+1

    if axis is not None:
        N = axis.gamma.shape[0]
        ma_ = np.zeros((axis.nfp*N+1, 3))
        ma0 = axis.gamma.copy()
        theta = 2*np.pi/axis.nfp
        rotmat = np.asarray([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]]).T

        for i in range(axis.nfp):
            ma_[(i*N):(((i+1)*N)), :] = ma0
            ma0 = ma0 @ rotmat
        ma_[-1, :] = axis.gamma[0, :]
        data = np.vstack((data, ma_))
        for i in range(ma_.shape[0]):
            labels.append('Magnetic Axis')
            groups.append(0)

    if extra_data is not None:
        for i, extra in enumerate(extra_data):
            labels += ['Extra %i' % i ] * extra.shape[0]
            groups += [-1-i] * extra.shape[0]
            data = np.vstack((data, extra)) 
    import plotly.express as px
    fig = px.line_3d(x=data[:,0], y=data[:,1], z=data[:,2],
                     color=labels, line_group=groups)
    fig.show()
