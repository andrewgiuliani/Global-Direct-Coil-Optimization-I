#!/usr/bin/env python
from pyplasmaopt import *
from scipy.optimize import minimize
import numpy as np
import os
import argparse
import sys
import shlex
import time
from qsc import Qsc

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--output",               default=   "", type=str,      help="any additional text for output folder")
parser.add_argument("--ppp",                  default=   20, type=int,      help="'points per period' for trapezoid rule on coils in Biot Savart law")
parser.add_argument("--Nt",                   default=    6, type=int,      help="'Number of terms' used in the Fourier expansion in coil/axis representation")
parser.add_argument("--nfp",                  default=    2, type=int,      help="number of field periods")
parser.add_argument("--nc-per-hp",            default=    4, type=int,      help="number of coils per half period")
parser.add_argument("--iota",                 default=  0.4, type=float,    help="target rotational transform")
parser.add_argument("--iota-weight",          default=   1., type=float,    help="target rotational transform weight")
parser.add_argument("--etabar-init",          default=   1., type=float,    help="initial etabar")
parser.add_argument("--clen",                 default=  4.5, type=float,    help="target coil length on each coil")
parser.add_argument("--clen-weight",          default=  1e1, type=float,    help="weight target coil length on each coil")
parser.add_argument("--max-kappa",            default=   5., type=float,    help="maximum acceptable coil curvature")
parser.add_argument("--max-kappa-w",          default=   1., type=float,    help="curvature weight")
parser.add_argument("--max-msc",              default=   5., type=float,    help="maximum mean squared curvature")
parser.add_argument("--max-msc-weight",       default=   1., type=float,    help="maximum mean squared curvature weight")
parser.add_argument("--min-dist",             default=  0.1, type=float,    help="minimum acceptable inter-coil distance")
parser.add_argument("--min-dist-weight",      default=  1e7, type=float,    help="minimum distance penalty term weight")
parser.add_argument("--mar",                  default=   1., type=float,    help="target mean magnetic axis radius")
parser.add_argument("--mar-weight",           default= 100., type=float,    help="magnetic axis radius weight")
parser.add_argument("--alen-weight",          default=1000., type=float,    help="arclength penalty term weight")
parser.add_argument("--id",                   default=    0, type=int,      help="id number")

with open(sys.argv[1]) as f:
    lines = f.readlines()
lines = shlex.split(lines[0])
args = parser.parse_args(lines)

np.random.seed(args.id)

outdir = f"/mnt/home/agiuliani/ceph/parameter_scan/publication_runs/l2g/naive/outputs/output-{args.id}/"
os.makedirs(outdir, exist_ok=True)

etabar = args.etabar_init
for outer_iter in range(10):
    print(f"OUTER ITER {outer_iter}")

    os.makedirs(outdir, exist_ok=True)
    info("Configuration: \n%s", args.__dict__)
    
    #increase the major radius in case the minor radius of the coils is longer than 1.
    mr = args.clen/(2*np.pi)
    Mr = 1. if mr < 1. else 1.1 * mr
    print(mr, Mr)
    (coils, ma, ma_ft, currents) = get_flat_data(Nt=args.Nt, Nt_ma=args.Nt, ppp=args.ppp, 
            nfp=args.nfp, ncoils_per_hp=args.nc_per_hp, 
            coil_major_radius= Mr, coil_minor_radius=mr)
    

    stellarator = CoilCollection(coils, currents, args.nfp, True)
    obj = SimpleNearAxisQuasiSymmetryObjective(
        stellarator, ma, ma_ft, args.iota, iota_weight=args.iota_weight, eta_bar=etabar,
        coil_length_target=args.clen, coil_length_weight=args.clen_weight, 
        magnetic_axis_radius_target=args.mar, magnetic_axis_radius_weight=args.mar_weight,
        target_curvature=args.max_kappa, curvature_weight=args.max_kappa_w, 
        msc_weight=args.max_msc_weight, msc_target=args.max_msc,
        arclength_weight=args.alen_weight, 
        minimum_distance=args.min_dist, distance_weight=args.min_dist_weight, 
        outdir=outdir)
    outdir = obj.outdir
    
    # initial callback
    if outer_iter == 0:
        x = obj.x.copy()
    if args.id > 0 and outer_iter == 0:
        scale = 0.01
        dofs_orig = x.copy()
        dofs_coil_orig = dofs_orig[obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1]]
        dofs_axis_orig = dofs_orig[obj.ma_dof_idxs[0]:obj.ma_dof_idxs[1]]
        Nperturb = 2
        while True:
            dofs_coil_perturb = dofs_coil_orig.copy()
            dofs_axis_perturb = dofs_axis_orig.copy()
            
            for i in range(len(obj.stellarator._base_coils)):
                m = obj.stellarator.dof_ranges[i][0]
                M = obj.stellarator.dof_ranges[i][1]
                coil_dofs = dofs_coil_perturb[m:M].reshape((3, 2*args.Nt+1)).copy()
                coil_dofs[:, :2*Nperturb+1] = coil_dofs[:, :2*Nperturb+1] + np.random.normal(0, scale, (3, 2*Nperturb+1))
                dofs_coil_perturb[m:M] = coil_dofs.flatten()
                #counter = 0
                #for i in range(3):
                #    self.coefficients[i][0] = dofs[counter]
                #    counter += 1
                #    for j in range(1, self.order+1):
                #        self.coefficients[i][2*j-1] = dofs[counter]
                #        counter += 1
                #        self.coefficients[i][2*j] = dofs[counter]
                #        counter += 1

            dofs_axis_perturb[:Nperturb+1] = dofs_axis_perturb[:Nperturb+1] + np.random.normal(0, scale, (Nperturb+1,))
            dofs_axis_perturb[args.Nt+1: args.Nt+1 +Nperturb] = dofs_axis_perturb[args.Nt+1: args.Nt+1 +Nperturb] + np.random.normal(0, scale, (Nperturb,))
            #counter = 0
            #for i in range(self.order+1):
            #    self.coefficients[0][i] = dofs[i]
            #for i in range(self.order):
            #    self.coefficients[1][i] = dofs[self.order + 1 + i]

            x = np.zeros(dofs_orig.shape)
            x[0] = np.random.normal(0.75, 0.1, 1)
            x[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] = dofs_orig[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] \
                    + np.random.normal(0., 0.3, dofs_orig[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]].shape) 
            x[obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1]]       = dofs_coil_perturb 
            x[obj.ma_dof_idxs[0]:obj.ma_dof_idxs[1]]           = dofs_axis_perturb
            
            obj.update(x)
            res = obj.res
            dres = obj.dres.copy()
 
            
            ln_coils = linking_number(stellarator.coils)
            
            # compute the axis linking number
            ln_axis = 0
            for c in stellarator._base_coils:
                ln_axis+= abs(linking_number([c, ma_ft])-1)
            hel = get_helicity(ma, x[0])
            
            print(f"INITIAL COIL LINKING NUMBER {ln_coils}, AXIS {ln_axis}, sigma solve {obj.qsf.sigma_success}, scale {scale}, hel {hel}")
            if (ln_coils > 0) or (ln_axis != 0) or (not obj.qsf.sigma_success) or (hel > 0):
                scale/=2
            else:
                break

    obj.update(x)
    obj.callback(x)
    
    maxiter = 5e3
    def J_scipy(x):
        obj.update(x)
        res = obj.res
        dres = obj.dres.copy()
        
        ln_coils = linking_number(stellarator.coils)
        
        # compute the axis linking number
        ln_axis = 0
        for c in stellarator._base_coils:
            ln_axis+= abs(linking_number([c, ma_ft])-1)
        hel = get_helicity(ma, x[0])

        print(f"COIL LINKING NUMBER {ln_coils}, AXIS {ln_axis}, sigma solve {obj.qsf.sigma_success}, helicity={hel}")
        if (not obj.qsf.sigma_success) or (ln_coils != 0) or (ln_axis != 0) or (hel > 0):
            res = obj.J_backup
            dres = -obj.dJ_backup.copy()
            print("triggering line search")
        
        return res, dres
    
    res = minimize(J_scipy, x, jac=True, method='bfgs', tol=1e-20,
                   options={"maxiter": maxiter},
                   callback=obj.callback)
    
    msc = [np.mean(c.kappa**2 * np.linalg.norm(c.dgamma_by_dphi, axis=-1))/np.mean(np.linalg.norm(c.dgamma_by_dphi, axis=-1)) for c in obj.stellarator._base_coils]
    iota_err = np.abs(obj.qsf.iota - obj.iota_target)/np.abs(obj.iota_target)
    curv_err = max(max([np.max(c.kappa) for c in obj.stellarator._base_coils]) - args.max_kappa, 0)/np.abs(args.max_kappa)
    msc_err = max(np.max(msc) - args.max_msc, 0)/np.abs(args.max_msc)
    min_dist_err = max(args.min_dist-obj.J_distance.min_dist(), 0)/np.abs(args.min_dist)
    alen_err = np.max([J.J() for J in obj.J_arclength])
    clen_err = np.max([np.abs((J_coil_length.J()-l)/l) for (i, (l, J_coil_length)) in enumerate(zip(obj.coil_length_targets, obj.J_coil_lengths))])

    print(f"OLD WEIGHTS iota {args.iota_weight} curv {args.max_kappa_w} min_dist {args.min_dist_weight} msc {args.max_msc_weight} alen {args.alen_weight} clen {args.clen_weight}")
    if iota_err > 0.001:
        args.iota_weight*=10
    if curv_err > 0.001:
        args.max_kappa_w*=10
    if min_dist_err > 0.001:
        args.min_dist_weight*=10
    if msc_err > 0.001:
        args.max_msc_weight*=10
    if alen_err > 0.001:
        args.alen_weight*=10
    if clen_err > 0.001:
        args.clen_weight*=10   
    print(f"NEW WEIGHTS iota {args.iota_weight} curv {args.max_kappa_w} min_dist {args.min_dist_weight} msc {args.max_msc_weight} alen {args.alen_weight} clen {args.clen_weight}\nERROR       iota {iota_err} curv {curv_err} min_dist {min_dist_err} msc {msc_err} alen {alen_err} clen {clen_err}\n")
    x = res.x.copy()
    etabar = obj.qsf.eta_bar
    x = x.copy()
    obj.set_dofs(x)
    obj.callback(x)

from simsopt._core import save
from simsopt.geo import curves_to_vtk as scurves_to_vtk

scoils = get_simsopt_coils(obj.stellarator._base_coils, obj.x[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]], args.nfp, True)
axis_curve = get_simsopt_curveRZ(ma)
axis = [axis_curve, obj.x[0]]

problem_config= {'iota_weight':args.iota_weight, 'iota_target': args.iota,
                 'curvature_weight':args.max_kappa_w, 'curvature_target': args.max_msc,
                 'min_dist_weight':args.min_dist_weight, 'min_dist_target': args.min_dist,
                 'msc_weight':args.max_msc_weight, 'msc_target': args.max_msc,
                 'alen_weight':args.alen_weight, 'alen_target': 0.001,
                 'clen_weight':args.clen_weight, 'clen_target': args.clen}
save([scoils, axis, problem_config, x, obj.qsf.sigma], outdir + f'nea_{args.id}.json')
