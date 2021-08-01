# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:07:17 2021

@author: jakee/pythonpadawanexe
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import ar_select_order
from scipy.sparse import load_npz
import modred as mr
import meshio
import os
import json
from scipy.signal import find_peaks
from matplotlib.patches import Ellipse
from sklearn import preprocessing

def main(filename: Path,Re,chronosi :int, start_n: int = 12000,end_n: int =15000,r:int = 11):

    W = load_npz(Path(str(os.path.dirname(__file__))+'\\'+'SIM_XDMF\\'
                               +"st08_navier_stokes_cylinder"+
                               "mass_Re_{}".format(str(Re).replace('.','-'))+".npz"))
    W = W.A
    Re = int(Re) 
       
    
    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()
        ssf = 1
           
        qT = np.array(
            [
                reader.read_data(k)[1]["vorticity"]
                for k in range(start_n, end_n,ssf)
            ]
        )
        
     
    xT = qT - qT.mean(0)  
    x  = xT.T
    
    n = end_n-start_n
    POD = mr.compute_POD_arrays_direct_method(x,inner_product_weights=W,mode_indices=range(r)) #,atol=1e-30
    MODES = POD.modes
    topos_mod = MODES
    chronos_mod = POD.proj_coeffs
    zeta = chronos_mod[chronosi]
    freqs = []
    for i in range(r):
        zeta = chronos_mod[i]
        dt = 0.001
        n_lags = 2 ** 1
        downsampling = 8#2**2
        lag_step = 2 **2
        p = n_lags * lag_step
        p = 2
        zeta = zeta[::downsampling]  #change zeta
        time = [*range(start_n,end_n)]
        time = time[::downsampling]
        step = np.arange(zeta.size) * downsampling 
        res =  AutoReg(endog=zeta,trend='c', lags=p).fit() #lag = lag_step * (1 + np.arange(n_lags))
        print(res.summary())
        asymptote = res.params[0] / (1 - res.params[1:].sum())
        print("asymptote:",asymptote)
        mu = 1 / res.roots
        print("Spectrum:",mu)
        s = np.log(mu) / (dt * downsampling * ssf)
        print("s:",s)
        modes = np.vander(mu,zeta.size,True).T
        modes /= np.linalg.norm(modes, axis = 0)
        coefficients = np.linalg.lstsq(modes, zeta - asymptote , rcond=None)[0]
        fundamental = np.argmax(abs(coefficients))
        sigma = s[fundamental].real
        omega = abs(s[fundamental].imag)
        period = 2 * np.pi / omega
        print("period:", period * dt)
        print("frequency:", 1 / (period * dt))
        print("decay time:", -1 / sigma, "time-steps")
        print("frequencies",abs(s.imag) / (2 * np.pi))
        fig,ax = subplots()
        ax.set_title(f'Re {Re} AR({p}) Chronos {i} with trend = "c"')
        ax.set_xlabel("snapshot")
        ax.set_ylabel('Vorticity')
        ax.plot(time, zeta, 
                marker=".", 
                linestyle="None", 
                color="green", 
                label="data"
                )
        ax.plot(time, asymptote + np.real_if_close(modes @ coefficients),
                color = "r",
                label = f"AR({p}) reconstruction",
                )
        
        '''
        ax.vlines( 
            np.arange(6) * period, 
            zeta.min(), 
            asymptote, 
            linestyle="dashed", 
            color="red", 
            label="AR({p}) period", 
        ) 
        
        '''
        
        
   
        if not os.path.exists(str(os.path.dirname(__file__))+'\POD_MODES'):
               try:
                   os.makedirs(str(os.path.dirname(__file__))+'\POD_MODES')
               except OSError as e:
                   if e.errno != errno.EEXIST:
                       raise
        if not os.path.exists(str(os.path.dirname(__file__))+'\POD_MODES\Re_{Re}.0'):
              try:
                  os.makedirs(str(os.path.dirname(__file__))+'\POD_MODES\Re_{Re}.0')
              except OSError as e:
                  if e.errno != errno.EEXIST:
                      raise
        fig.savefig(str(os.path.dirname(__file__))+"\POD_MODES\Re_{}.0\AR({})_Chronos_{}_Time_{}-{}.png".format(Re,lag_step,i,start_n,end_n))
        ax.legend()
        freqs.append(mu[fundamental])
        
    return coefficients,np.array(freqs),p
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    U =  1.4994965504069229 #np.max(uv0[inlet_dofs]) , can a
   # nu =  (0.1*U)/np.arange(81,86,1)
    nu =  np.unique(((0.1*U)/np.arange(150,151,10)))
    #nu =  np.unique(((0.1*U)/np.arange(70,81,1)))
    #nu = [0.1*U/118]
    Re_ls = []
    GRATES_CHRON1 = []
    FREQS_CHRON1 = []
    GRATES_CHRON2 = []
    FREQS_CHRON2 = []
    ROOTS_CHRON1 = []
    ROOTS_CHRON2 = []
    #U_avg = 1
    from cylinder import radius as rad
    Diam = 2*rad
    
    
    for i in range(len(nu)):
        Re = round((U*Diam)/nu[i],4)
        Re_ls.append(Re)
        parser = ArgumentParser()
        parser.add_argument(
            "-f",
            "--filename",
            type=Path,
            default=Path(str(os.path.dirname(__file__))+'\\'+'SIM_XDMF\\'
                                   +"st08_navier_stokes_cylinder"+
                                   "_Re_{}.xdmf".format(str(Re).replace('.','-'))),
        )
        args = parser.parse_args()
        print(args)
        chronosi = 0
        coefficients,mu,p = main(Path(args.filename),Re,chronosi)
        if ('coefficients_matrix' not in locals()):
            coefficients_matrix = coefficients.T
        else:
           coefficients_matrix = np.vstack((coefficients_matrix,coefficients.T))
        if ('mu_matrix' not in locals()):
            mu_matrix = mu.T
        else:
           mu_matrix = np.vstack((mu_matrix,mu.T))
            
                 
    '''     
    for i in range(coefficients_matrix.shape[0]):
        theta = np.arange(0,100,1)*2*np.pi/100
        coeff_row = coefficients_matrix[i]
        mu_row = mu_matrix[i]
        mu_row = mu_row
        ells = [Ellipse(xy=(mu.real,mu.imag),
                width=coeff.real/np.abs(coeff), height=coeff.imag/np.abs(coeff),
                angle=0)
                for mu,coeff in zip(mu_row, coeff_row)]  
        fig,ax = subplots()
        ax.set_xlim(left=-2,right=2)
        ax.set_ylim(bottom=-2,top=2)      
        for e in ells:
            # if e.height < 0:
            #     e.set_alpha(alpha=0.5)
            ax.add_artist(e)
            
        ax.set(aspect=1)
        ax.plot(np.sin(theta),np.cos(theta),'--k')
        ax.set_title(f'Re {Re_ls[i]} AR({p}) with trend = "c" Roots')
        ax.set_ylabel(f"Im (μ)")
        ax.set_xlabel(f"Re (μ)")
           
        fig.savefig(str(os.path.dirname(__file__))+f'\Re {Re_ls[i]} AR({p}) with trend = c Roots & Coefficients.png')
    '''
    for i in range(len(nu)):
        theta = np.arange(0,100,1)*2*np.pi/100
        if len(nu) > 1:
            mu_row = mu_matrix[i]
        else:
            mu_row = mu_matrix
        fig,ax = subplots()
        ax.set(aspect=1)
        ax.set_xlim(left=-2,right=2)
        ax.set_ylim(bottom=-2,top=2)      
        ax.scatter(mu_row.real,mu_row.imag,marker='o')
        
        ax.plot(np.sin(theta),np.cos(theta),'--k')
        ax.set_title(f'Re {Re_ls[i]} AR({p}) with trend = "c" Roots')
        ax.set_ylabel(f"Im (μ)")
        ax.set_xlabel(f"Re (μ)")
           
        fig.savefig(str(os.path.dirname(__file__))+f'\Re {Re_ls[i]} AR({p}) with trend = c Roots.png')   