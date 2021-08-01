# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:53:52 2021

@author: jakee
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
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq , ifft
from numpy import random
import json
from matplotlib.ticker import FormatStrFormatter
def do_FFT(downsampling,zeta: np.ndarray,count: int,string: str,sr: int = 1000,):
    sr = sr/downsampling
    zeta = zeta - zeta.mean(0)
    FFT = fft(zeta)
    FFT_FREQ = fftfreq(zeta.size,1/sr)
    X_oneside = FFT
    f_oneside = FFT_FREQ
    n = 2 ** 5
    X_oneside = X_oneside[:len(X_oneside)//n]
    f_oneside = f_oneside[:len(f_oneside)//n]
    '''
    fig,ax = plt.subplots()
    ax.stem(f_oneside[:int(len(f_oneside)/2)], np.abs(X_oneside[:int(len(f_oneside)/2)]), 'b', 
            )
    ax.set(xlabel='Freq (rad/s)',ylabel='Normalized FFT Amplitude |X(freq)|',title="FFT of Re {}".format(Re)+string)
    fig.show()
    fig.savefig("FFT of Re {}"+string+ ".png".format(Re))
    '''
    kmax = np.argmax(np.abs(X_oneside))
    max_f = f_oneside[kmax]
    print(max_f)
    return X_oneside ,f_oneside
def chronos_AR_P(downsampling,chronos_mod: np.ndarray,lags_p: int =2):
    zeta = chronos_mod
    # r = chronosmode
    # t = time_window[1]
    
    
    zeta = zeta[::downsampling]  #change zeta
    step = np.arange(zeta.size) * downsampling 
    res =  AutoReg(endog=zeta, lags=lags_p,old_names=False).fit()
    
    print(res.summary())
    estcoeff = res.params #estimated model coefficients
    print("Estimated Model Coefficients:",estcoeff)
    mu = res.roots**-1        # roots
    print("Roots:",mu)
    s = mu 
    print("s:", s) 
    omega = np.angle(s[0])/(2*np.pi*downsampling)#s[0].imag 
    period = np.abs(1 / omega) 
    print("period:", period, " time-steps") 
    sigma = np.log(np.abs(s[0]))/ downsampling #s[0].real 
    print("decay time:", np.abs(1 / sigma), " time-steps") 
    asymptote = estcoeff[0] / (1 - estcoeff[1:].sum()) 
    
    modes = np.vander(mu,zeta.size,True).T
    modes /= np.linalg.norm(modes, axis =0)
    coefficients = np.linalg.lstsq(modes, zeta - asymptote , rcond=None)[0]
   
    #see Intertemporal effect of shocks 
    #https://en.wikipedia.org/wiki/Autoregressive_model

        
   
    kmax = np.argmax(zeta) 
    amplitude = zeta[kmax] - asymptote 
    
    reconstruction =   asymptote + np.real_if_close(modes @ coefficients)
    return step,reconstruction
    
    
def get_cmap(n, name='tab10'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def main(filename: Path,Re, start_n: int = 10000,end_n: int =15000,r:int = 8):

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
    chronos_mod =  POD.proj_coeffs
    zeta = chronos_mod[0]
    downsampling = 1
    step = np.arange(zeta.size)
    
    r = 8
    v = POD.proj_coeffs[:r, ::downsampling]
    downsampling = 16
    fig, axes = subplots(r // 2, sharex=True,dpi=400)
    fig.set_aspect = 'auto'
    zoom = 2.5
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    lags_p =2
    for count,i in enumerate(v):
        zeta = i
        zeta = zeta - zeta.mean(0)
        X ,Y = chronos_AR_P(downsampling,zeta,lags_p)
        
        line1, = axes[count // 2].plot(X,Y,marker='x',linestyle="None",label=None if count else f"AR({lags_p})")
        line2, = axes[count // 2].plot(step,zeta,label=None if count else "chronos")
        axes[count//2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[count//2].set_ylabel("Chronos Pair {}".format(count//2 + 1))
        if count % 2 == 0 :
            line1.set_color('C0')
            line2.set_color('C0')
        else:
            line1.set_color("darkorange")
            line2.set_color("darkorange")
        axes[count//2].set_xticklabels([int(ii) for ii in (axes[count//2].get_xticks()+start_n)])
    axes[-1].set(xlabel = 'Snapshot')
    axes[0].legend() 
    fig.show()
    fig.savefig("AR(2) of {} Chronos Pairs at Re {}.png".format(r//2,Re))
    # return X_oneside ,f_oneside

if __name__ == "__main__":
    from argparse import ArgumentParser
    U =  1.4994965504069229 #np.max(uv0[inlet_dofs]) , can a
    # nu =  (0.1*U)/np.arange(81,86,1)
    nu =  np.unique(((0.1*U)/np.arange(150,151,10))) # (90,151,10)
    # nu = [0.1*U/100]
    Re_ls = []
    FREQS = []
    AMPS = []
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
        main(Path(args.filename),Re)
   
    # fig.show()
    # fig.savefig("FFT of First Chronos Mode.png")
    