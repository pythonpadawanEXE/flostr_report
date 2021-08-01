# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:07:17 2021

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
import json
from scipy.signal import find_peaks

def main(filename: Path,Re,chronosi :int, start_n: int = 3000,end_n: int =15000,r:int = 1):

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
    
   
    #df_stationarityTest = adfuller(zeta, autolag='BIC')
    #print("P-value: ", df_stationarityTest[1])
    # pacf = plot_pacf(zeta, lags=100 , title ="Partial Autocorrelation @ Re ={}".format(Re))
    
    downsampling = 1
    
    zeta = zeta[::downsampling]  #change zeta
    step = np.arange(zeta.size) * downsampling 
    res =  AutoReg(endog=zeta, lags=2,old_names=False).fit()
    
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
    asymptote = estcoeff[0] / (1 - estcoeff[1:].sum()) # see Intertemporal effect of shocks https://en.wikipedia.org/wiki/Autoregressive_model
    
    dt = 1
    t = end_n - start_n
    Uyp1 = chronos_mod[0]
    smoothed = np.zeros(t)
    smoothed[0]=Uyp1[0]
    alpha = .03 * dt
    for i in range(1,t):
    	smoothed[i]=(1-alpha)*smoothed[i-1]+alpha*Uyp1[i-1]
        
   
    kmax = np.argmax(zeta) 
    amplitude = zeta[kmax] - asymptote 
    
    
    
    fig, ax = subplots() 
    ax.plot(step, zeta, marker=".", linestyle="None", color="green", label="data") 
    ax.set_xlabel("relative time-step") 
    ax.set_ylabel("vorticity") 
    fig.suptitle("Chronos Mode Steady State for Mode {}  @ Re ={}".format(r,Re))
    # ax.plot( 
    #     step, 
    #     asymptote + amplitude * np.exp(sigma * (step - kmax * downsampling)),
    #     linestyle="dotted", 
    #     color="k", 
    #     label="AR(2) envelope", 
    # ) 
    
    ax.vlines( 
        np.arange(6) * period, 
        zeta.min(), 
        asymptote, 
        linestyle="dashed", 
        color="red", 
        label="AR(2) period", 
    ) 
    ax.legend() 
    fig.savefig("Chronos Mode Steady State for Mode {}  @ Re ={} Entire Range.png".format(r,Re))

    return sigma,omega,s[0]

if __name__ == "__main__":
    from argparse import ArgumentParser
    U =  1.4994965504069229 #np.max(uv0[inlet_dofs]) , can a
   # nu =  (0.1*U)/np.arange(81,86,1)
    nu =  np.unique(((0.1*U)/np.arange(70,81,1)))
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
    for ii in range(2):
        
        chronosi = ii
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
            GRATE,FREQ,ROOT = main(Path(args.filename),Re,chronosi)
            if chronosi == 0:
                GRATES_CHRON1.append(GRATE)
                FREQS_CHRON1.append(FREQ)
                ROOTS_CHRON1.append(ROOT)
            elif chronosi ==1:
                GRATES_CHRON2.append(GRATE)
                FREQS_CHRON2.append(FREQ)
                ROOTS_CHRON2.append(ROOT)
            else:
                 print("Error!")
        if not os.path.exists(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'):
            try:
                os.makedirs(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if chronosi == 0:
            with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\GRATES_CHRON1.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(GRATES_CHRON1, f, indent=2) 
            with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\FREQS_CHRON1.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(FREQS_CHRON1, f, indent=2) 
            # with open(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'+'\ROOTS_CHRON1.json', 'w') as f:
            #         # indent=2 is not needed but makes the file human-readable
            #         json.dump(ROOTS_CHRON1, f, indent=2) 
            save = open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\ROOTS_CHRON1.dat',"w")
            np.savetxt(save, ROOTS_CHRON1 , newline = "\r\n")
            save.close()
           
        elif chronosi == 1:
            with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\GRATES_CHRON2.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(GRATES_CHRON2, f, indent=2) 
            with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\FREQS_CHRON2.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(FREQS_CHRON2, f, indent=2) 
            # with open(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'+'\ROOTS_CHRON2.json', 'w') as f:
            #         # indent=2 is not needed but makes the file human-readable
            #         json.dump(ROOTS_CHRON2, f, indent=2)
            save = open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\ROOTS_CHRON2.dat',"w")
            np.savetxt(save, ROOTS_CHRON2 , newline = "\r\n")
            save.close()
            
        else:
            print("Error!")
        with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\Re_ls.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(Re_ls, f, indent=2) 
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
        # fig,ax = plt.subplots()
        # ax.scatter(Re_ls,FREQS)
        # ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Frequency')
        # ax.grid()
        # #plt.legend()
        # fig.suptitle("Frequency of First Chronos Mode Steady State")
        # fig.savefig("Frequency of First Chronos Mode from AR(2) Steady State")
        # plt.show()    
        
        
        
        # fig,ax = plt.subplots()
        # ax.scatter(Re_ls,GRATES)
        # ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Growth Rate')
        # ax.grid()
        # #plt.legend()
        # fig.suptitle("Growth Rate  of First Chronos Mode")
        # fig.savefig("Supercritical Hopf Bifurcation Growth Rate Transition of First Chronos Mode")
        # plt.show()
      
