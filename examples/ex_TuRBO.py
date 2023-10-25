#!/usr/bin/env python
from pyplasmaopt import *
from scipy.optimize import minimize
import numpy as np
import os
import argparse
import sys
import shlex
import time

# DEFAULT PARAMETERS TO USE IN THE OPTIMIZATION
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--output",               default=   "", type=str,      help="any additional text for output folder")
parser.add_argument("--ppp",                  default=   60, type=int,      help="'points per period' for trapezoid rule on coils in Biot Savart law")
parser.add_argument("--Nt",                   default=    2, type=int,      help="'Number of terms' used in the Fourier expansion in coil/axis representation")
parser.add_argument("--Nt_ma",                default=    2, type=int,      help="'Number of terms' used in the Fourier expansion in coil/axis representation")
parser.add_argument("--nfp",                  default=    2, type=int,      help="number of field periods")
parser.add_argument("--nc-per-hp",            default=    4, type=int,      help="number of coils per half period")
parser.add_argument("--iota",                 default=  0.4, type=float,    help="target rotational transform")
parser.add_argument("--iota-weight",          default=   1e2, type=float,    help="target rotational transform weight")
parser.add_argument("--etabar-init",          default=   1., type=float,    help="initial etabar")
parser.add_argument("--clen",                 default=  4.5, type=float,    help="target coil length on each coil")
parser.add_argument("--clen-weight",          default=   1e2, type=float,    help="weight target coil length on each coil")
parser.add_argument("--max-kappa",            default=   5., type=float,    help="maximum acceptable coil curvature")
parser.add_argument("--max-kappa-w",          default=   1e2, type=float,    help="curvature weight")
parser.add_argument("--max-msc",              default=   5., type=float,    help="maximum mean squared curvature")
parser.add_argument("--max-msc-weight",       default=   1e2, type=float,    help="maximum mean squared curvature weight")
parser.add_argument("--min-dist",             default=  0.1, type=float,    help="minimum acceptable inter-coil distance")
parser.add_argument("--min-dist-weight",      default=  1e8, type=float,    help="minimum distance penalty term weight")
parser.add_argument("--mar",                  default=   1., type=float,    help="target mean magnetic axis radius")
parser.add_argument("--mar-weight",           default= 100., type=float,    help="magnetic axis radius weight")
parser.add_argument("--alen-weight",          default=   1, type=float,    help="arclength penalty term weight")
parser.add_argument("--id",                   default=    0, type=int,      help="id number")

with open(sys.argv[1]) as f:
    lines = f.readlines()
lines = shlex.split(lines[0])
args = parser.parse_args(lines)


outdir = f"/mnt/home/agiuliani/ceph/parameter_scan/publication_runs/l2g/sector/outputs/output-{args.id}/"
os.makedirs(outdir, exist_ok=True)

etabar = args.etabar_init

os.makedirs(outdir, exist_ok=True)
info("Configuration: \n%s", args.__dict__)


(coils, ma, ma_ft, currents) = get_flat_data(Nt=args.Nt, Nt_ma=args.Nt_ma, ppp=args.ppp, nfp=args.nfp, ncoils_per_hp=args.nc_per_hp)
stellarator = CoilCollection(coils, currents, args.nfp, True)
obj = SimpleNearAxisQuasiSymmetryObjective(
    stellarator, ma, ma_ft, args.iota, iota_weight=args.iota_weight, eta_bar=etabar,
    coil_length_target=args.clen, coil_length_weight=args.clen_weight, 
    magnetic_axis_radius_target=args.mar, magnetic_axis_radius_weight=args.mar_weight,
    target_curvature=args.max_kappa, curvature_weight=args.max_kappa_w, 
    msc_weight=args.max_msc_weight, msc_target=args.max_msc,
    arclength_weight=args.alen_weight, 
    minimum_distance=args.min_dist,  distance_weight=args.min_dist_weight, 
    outdir=outdir)
outdir = obj.outdir

# initial callback
x = obj.x.copy()
obj.update(x)
obj.callback(x)
print("OPTIMIZATION PROBLEM SIZE ", x.size)


class obj_TuRBO:
    def __init__(self, dim, Nt, Nt_ma):
        self.Nt = Nt
        self.Nt_ma = Nt_ma
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.zeros(dim)
        
        # bounds on etabar
        self.lb[0] = 0.
        self.ub[0] = 2.
        
        #bounds on mu0*currents
        self.lb[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] = -1.
        self.ub[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] =  1.
        
        R0 = obj.ma_dof_idxs[0]
        R1 = R0 + args.Nt_ma + 1
        Z0 = R1
        Z1 = Z0 + args.Nt_ma
        
        # next (2*N+1) dofs are R dofs
        self.lb[R0:R1] = -1/(args.nfp**2+1)
        self.ub[R0:R1] =  1/(args.nfp**2+1)
        # next (2*N+1) dofs are Z dofs
        self.lb[Z0:Z1] = -0.2
        self.ub[Z0:Z1] =  0.2

        Rminor = args.clen / (2*np.pi)
        overlap_factor = 1.1
        dt = overlap_factor/(2*args.nfp*args.nc_per_hp)
        counter = obj.coil_dof_idxs[0]
        
        # next 3*(2*N+1) dofs are XYZ coil dofs
        for i in range(args.nc_per_hp):
            self.lb[counter+0*(2*args.Nt+1):counter+1*(2*args.Nt+1)] = 0.
            self.lb[counter+1*(2*args.Nt+1):counter+2*(2*args.Nt+1)] = -dt*0.5
            self.lb[counter+2*(2*args.Nt+1):counter+3*(2*args.Nt+1)] = -Rminor
            
            self.ub[counter+0*(2*args.Nt+1):counter+1*(2*args.Nt+1)] = 1.+Rminor + np.max([Rminor-1., 0])
            self.ub[counter+1*(2*args.Nt+1):counter+2*(2*args.Nt+1)] = +dt*0.5
            self.ub[counter+2*(2*args.Nt+1):counter+3*(2*args.Nt+1)] = +Rminor
            counter+=3*(2*args.Nt+1)
        
        self.minx = np.zeros((self.dim,))
        self.min_res = np.inf
        
        order = args.Nt

        # matrix for projecting anchor points onto Fourier basis
        A = np.zeros((2*order+1,2*order+1))
        t = np.linspace(0, 1, 2*order+1, endpoint=False)
        freq = np.arange(args.Nt+1)
        
        A[::2,:]= np.cos(2*np.pi*freq[:,None]*t[None,:])
        A[1::2,:]= np.sin(2*np.pi*freq[-order:,None]*t[None,:]) 
        self.A = 2/(2*order+1)*A
        self.A[0, :]/=2
        
        order = args.Nt_ma
        A = np.zeros((2*order+1,2*order+1))
        t = np.linspace(0, 1, 2*order+1, endpoint=False)
        freq = np.arange(args.Nt_ma+1)
        
        A[::2,:]= np.cos(2*np.pi*freq[:,None]*t[None,:])
        A[1::2,:]= np.sin(2*np.pi*freq[-order:,None]*t[None,:]) 
        self.A_ma = 2/(2*order+1)*A
        self.A_ma[0, :]/=2

    
    def XYZ2Fourier(self, xyz):
        # convert the anchor points to Fourier coefficients

        A = self.A
        A_ma = self.A_ma
        fourier =  np.zeros(xyz.shape)
        
        # etabar
        fourier[0] = xyz[0]

        RZpts = xyz[obj.ma_dof_idxs[0]:obj.ma_dof_idxs[1]]
        
        # R dofs
        Rdofs = np.concatenate((RZpts[:args.Nt_ma+1], RZpts[1:args.Nt_ma+1][::-1]))
        Rdofs -= np.mean(Rdofs)
        Rdofs += 1.
        Rc = (A_ma@Rdofs)[::2]
        
        # Z dofs
        Zdofs = np.concatenate(([0], RZpts[args.Nt_ma+1:], -RZpts[args.Nt_ma+1:][::-1]))
        Zs = (A_ma@Zdofs)[1::2]
        fourier[obj.ma_dof_idxs[0]:obj.ma_dof_idxs[1]] = np.concatenate([Rc, Zs])

        # currents
        fourier[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] = xyz[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]]
        
        RTZ_pts = xyz[obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1]]
        counter_global = obj.coil_dof_idxs[0]
        counter_local = 0

        for i in range(args.nc_per_hp):
            RTZpts = RTZ_pts[counter_local:counter_local+3*(2*args.Nt+1)].reshape((3, -1))
            
            # sort counter clockwise in the RZ plane
            theta = np.arctan2(RTZpts[2] - np.mean(RTZpts[2]), RTZpts[0] - np.mean(RTZpts[0]))
            idx = np.argsort(theta)
            RTZpts = RTZpts[:, idx]
            
            R = RTZpts[0, :]
            T = RTZpts[1, :]
            Z = RTZpts[2, :]
            XYZpts = np.concatenate((R[None, :] * np.cos(2*np.pi * T[None, :]), R[None, :] * np.sin(2*np.pi * T[None, :]), Z[None, :]), axis=0)

            # rotate ...
            angle = (i+0.5)*2*np.pi/args.nfp/2./args.nc_per_hp
            RXYZpts = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])@ XYZpts
            Xpts = RXYZpts[0]
            Ypts = RXYZpts[1]
            Zpts = RXYZpts[2]
            
            Xsc = A@Xpts
            Ysc = A@Ypts
            Zsc = A@Zpts
            
            # sanity check
            #tvals = np.linspace(0, 2*np.pi, 5, endpoint=False)[:, None]
            #modes = np.concatenate([np.ones((5,1)), np.sin(1*tvals), np.cos(1*tvals), np.sin(2*tvals), np.cos(2*tvals)], axis=1)
            #import ipdb;ipdb.set_trace()

            fourier[counter_global:counter_global+3*(2*args.Nt+1)] = np.concatenate([Xsc, Ysc, Zsc])
            counter_global += 3*(2*args.Nt+1)
            counter_local += 3*(2*args.Nt+1)

        return fourier

    def J_TuRBO(self, x):
        obj.update(x)
        res = obj.res
        dres = obj.dres.copy()
        
        ln_coils = linking_number(stellarator.coils)
        
        # compute the axis linking number
        ln_axis = 0
        for c in stellarator._base_coils:
            ln_axis+= abs(linking_number([c, ma_ft])-1)
        hel = get_helicity(ma, x[0])
    
        if not obj.qsf.sigma_success:
            res = 1e8
        elif ln_coils > 0:
            res+= 1e7 * ln_coils
        elif ln_axis != 0:
            res+= 1e7*ln_axis
        elif hel > 0.:
            res+= 1e7*hel
    
        msg=f"ln_coils {ln_coils}, ln_axis {ln_axis}, helicity {hel}, {obj.qsf.sigma_success}"
        return res, dres, msg

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        fourier = self.XYZ2Fourier(x)
        res, _, success = self.J_TuRBO(fourier)

        res_dict={}
        res_dict['nonQS'] = obj.res1
        res_dict['coil length'] = obj.res2
        res_dict['iota'] = obj.res4
        res_dict['curvature'] = obj.res5
        res_dict['msc'] = obj.res6
        res_dict['arclength'] = obj.res8
        res_dict['min dist'] = obj.res9
        print(f"{res:.2e}", [f"{k}: {res_dict[k]:.1e}" for k in res_dict.keys()], f"{obj.qsf.iota:.1f}", [f"{J.J():.1f}" for J in obj.J_coil_lengths], success)

        if res < self.min_res:
            self.min_res = res
            self.minx[:] = x
            np.savetxt(outdir+'global_opt_minx.txt', x)
        return res

f = obj_TuRBO(obj.x.size, args.Nt, args.Nt_ma)

from turbo import Turbo1
turbo_m = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=1000,  # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=15000,  # Maximum number of evaluations
    batch_size=100,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo_m.optimize()
X = turbo_m.X  # Evaluated points
fX = turbo_m.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]
print("initial global optimization is done!")

out = f(x_best.copy())
res = obj.res
xfourier_best = obj.x.copy()

eb = xfourier_best[0]
RZ = xfourier_best[obj.ma_dof_idxs[0]:obj.ma_dof_idxs[1]]
R = RZ[:f.Nt_ma+1]
Z = RZ[f.Nt_ma+1:]
coils_dof    = xfourier_best[obj.coil_dof_idxs[0]:obj.coil_dof_idxs[1]]
currents_dof = xfourier_best[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]]

# IN PHASE I, CHANGE THE DEFAULT PARAMETERS TO USE IN THE OPTIMIZATION
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--output",               default=   "", type=str,      help="any additional text for output folder")
parser.add_argument("--ppp",                  default=   20, type=int,      help="'points per period' for trapezoid rule on coils in Biot Savart law")
parser.add_argument("--Nt",                   default=    6, type=int,      help="'Number of terms' used in the Fourier expansion in coil/axis representation")
parser.add_argument("--Nt_ma",                default=    6, type=int,      help="'Number of terms' used in the Fourier expansion in coil/axis representation")
parser.add_argument("--nfp",                  default=    2, type=int,      help="number of field periods")
parser.add_argument("--nc-per-hp",            default=    4, type=int,      help="number of coils per half period")
parser.add_argument("--iota",                 default=  0.4, type=float,    help="target rotational transform")
parser.add_argument("--iota-weight",          default=   1e2, type=float,    help="target rotational transform weight")
parser.add_argument("--etabar-init",          default=   1., type=float,    help="initial etabar")
parser.add_argument("--clen",                 default=  4.5, type=float,    help="target coil length on each coil")
parser.add_argument("--clen-weight",          default=   1e2, type=float,    help="weight target coil length on each coil")
parser.add_argument("--max-kappa",            default=   5., type=float,    help="maximum acceptable coil curvature")
parser.add_argument("--max-kappa-w",          default=   1e2, type=float,    help="curvature weight")
parser.add_argument("--max-msc",              default=   5., type=float,    help="maximum mean squared curvature")
parser.add_argument("--max-msc-weight",       default=   1e2, type=float,    help="maximum mean squared curvature weight")
parser.add_argument("--min-dist",             default=  0.1, type=float,    help="minimum acceptable inter-coil distance")
parser.add_argument("--min-dist-weight",      default=  1e8, type=float,    help="minimum distance penalty term weight")
parser.add_argument("--mar",                  default=   1., type=float,    help="target mean magnetic axis radius")
parser.add_argument("--mar-weight",           default= 100., type=float,    help="magnetic axis radius weight")
parser.add_argument("--alen-weight",          default=   1, type=float,    help="arclength penalty term weight")
parser.add_argument("--id",                   default=    0, type=int,      help="id number")
args = parser.parse_args(lines)

# INCREASE THE NUMBER OF FOURIER HARMONICS THAT REPRESENT THE AXIS AND COILS HERE
x = np.zeros(1 + (2*args.Nt_ma+1) + args.nc_per_hp +  3*(2*args.Nt + 1)*args.nc_per_hp)
count = 0
x[count] = eb; count+=1
for i in range(R.size):
    x[count+i] = R[i]
count+=args.Nt_ma+1
for i in range(Z.size):
    x[count+i] = Z[i]
count+=args.Nt_ma
for i in range(args.nc_per_hp):
    x[count+i] = currents_dof[i]
count+=args.nc_per_hp
for i in range(args.nc_per_hp):
    for j in range(3):
        for k in range(2*f.Nt + 1):
            x[count+k] = coils_dof[3*i*(2*f.Nt+1) + j*(2*f.Nt+1)+ k]
        count+=2*args.Nt+1

with open(sys.argv[1]) as ff:
    lines = ff.readlines()
lines = shlex.split(lines[0])
args = parser.parse_args(lines)

# run the penalty method
for outer_iter in range(10):
    print(f"OUTER ITER {outer_iter}")

    os.makedirs(outdir, exist_ok=True)
    info("Configuration: \n%s", args.__dict__)
    
    (coils, ma, ma_ft, currents) = get_flat_data(Nt=args.Nt, Nt_ma=args.Nt_ma, ppp=args.ppp, 
            nfp=args.nfp, ncoils_per_hp=args.nc_per_hp)
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
    
    obj.update(x)
    obj.callback(x)
    
    maxiter = 5e3
    def J_scipy(x):
        obj.update(x)
        res = obj.res
        dres = obj.dres.copy()
        
        ln_coils = linking_number(stellarator.coils)
        ln_axis = 0
        for c in stellarator._base_coils:
            ln_axis+= abs(linking_number([c, ma_ft])-1)
        hel = get_helicity(ma, x[0])

        print(f"ln_coils {ln_coils}, ln_axis {ln_axis}, helicity {hel}, {obj.qsf.sigma_success}")
        if (not obj.qsf.sigma_success) or (ln_coils != 0) or (ln_axis != 0) or (hel > 0):
            res = obj.J_backup
            dres = -obj.dJ_backup.copy()
            print("triggering line search")
        
        return res, dres
    
    res = minimize(J_scipy, x, jac=True, method='bfgs', tol=1e-20,
                   options={"maxiter": maxiter},
                   callback=obj.callback)
    
    print("%s" % res)
    
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


# convert to simsopt coils
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
save([scoils, axis, etabar], outdir + f'nae_{args.id}.json')
