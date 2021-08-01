# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:46:34 2021

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

def main(filename: Path,Re, start_n: int = 3000,end_n: int =15000,probetype= None,vorticity = None):

    Re = int(Re) 
       
    
    downsampling = 1
    zeta = vorticity[start_n:end_n]
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
    omega =  np.angle(s[0])/(2*np.pi*downsampling) 
    period =  np.abs(1 / omega) #1/omega #np.abs(2 * np.pi / omega) 
    print("period:", period, " time-steps") 
    sigma = np.log(np.abs(s[0]))/ downsampling #s[0].real 
    print("decay time:", np.abs(1 / sigma), " time-steps") 
    asymptote = estcoeff[0] / (1 - estcoeff[1:].sum()) # see Intertemporal effect of shocks https://en.wikipedia.org/wiki/Autoregressive_model
    
    dt = 1
    t = end_n - start_n
    Uyp1 = zeta
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
    fig.suptitle("Eventual Amplitude AR(2) {}  @ Re ={}".format(probetype,Re))
    ax.plot( 
        step, 
        asymptote + amplitude * np.exp(sigma * (step - kmax * downsampling)),
        linestyle="dotted", 
        color="k", 
        label="AR(2) envelope", 
    ) 
    
    ax.vlines( 
        np.arange(6) * period, 
        zeta.min(), 
        asymptote, 
        linestyle="dashed", 
        color="red", 
        label="AR(2) period", 
    ) 
    ax.legend() 
    fig.savefig("Eventual Amplitude AR(2) {}  @ Re ={} Entire Range.png".format(probetype,Re))

    return sigma,omega

if __name__ == "__main__":
    from argparse import ArgumentParser
    for ii in range(2):
            chronosi = ii
            U =  1.4994965504069229 #np.max(uv0[inlet_dofs]) , can a
            nu =  np.unique(((0.1*U)/np.arange(70,81,1)))
            #nu = [0.1*U/118]
            Re_ls = []
            GRATES_AFT = []
            GRATES_FORE = []
            FREQS_AFT = []
            FREQS_FORE = []
            #U_avg = 1
            from cylinder import radius as rad
            Diam = 2*rad
    
    
            if ii == 0:
                probetype = "Fore Vorticity Probe"
            if ii == 1:    
                probetype = "Aft Vorticity Probe"
        
            for i in range(len(nu)):
                Re = round((U*Diam)/nu[i],4)
                Re_ls.append(int(Re))
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
                # print(args)
                
                
                
                
                
                filename= Path(args.filename)
                probes = np.array([4, 5])
                with meshio.xdmf.TimeSeriesReader(filename) as reader:
            
                    points, _ = reader.read_points_cells()
                    print("Points:", points[probes, :2])
            
                    time = []
                    vorticity = []
                    steps = reader.num_steps
                    for k in range(steps):
                        t, pd, _ = reader.read_data(k)
                        time.append(t)
                        vorticity.append(pd["vorticity"][probes])
            
                vorticity = np.array(vorticity).T
                
                vorticity = vorticity[ii]
                
                
                
                
                
                
                
                GRATE,FREQ = main(Path(args.filename),Re,probetype=probetype,vorticity=vorticity)
                if ii == 0:
                    GRATES_FORE.append(GRATE)
                    FREQS_FORE.append(FREQ)
                elif ii == 1: 
                    GRATES_AFT.append(GRATE)
                    FREQS_AFT.append(FREQ)
                else:
                    print("Error!")
            if not os.path.exists(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'):
                try:
                    os.makedirs(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON')
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            if chronosi == 0:
                with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\GRATES_FORE.json', 'w') as f:
                        # indent=2 is not needed but makes the file human-readable
                        json.dump(GRATES_FORE, f, indent=2) 
                with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\FREQS_FORE.json', 'w') as f:
                        # indent=2 is not needed but makes the file human-readable
                        json.dump(FREQS_FORE, f, indent=2) 
                
            elif chronosi == 1:
                with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\GRATES_AFT.json', 'w') as f:
                        # indent=2 is not needed but makes the file human-readable
                        json.dump(GRATES_AFT, f, indent=2) 
                with open(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON'+'\FREQS_AFT.json', 'w') as f:
                        # indent=2 is not needed but makes the file human-readable
                        json.dump(FREQS_AFT, f, indent=2) 
            else:
                print("Error!")
                
         
            
    # fig,ax = plt.subplots()
    # ax.scatter(Re_ls,FREQS_FORE,label="Fore Probe",marker='+')
    # ax.scatter(Re_ls,FREQS_AFT,label="Aft Probe",marker='x')
    # ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Frequency')
    # ax.grid()
    # plt.legend()
    # fig.suptitle("Steady State Supercritical Frequency AR(2)")
    # fig.savefig("Steady State Supercritical Frequency AR(2)")
    # plt.show()    
    
    
    
    # fig,ax = plt.subplots()
    # ax.scatter(Re_ls,GRATES_FORE,label="Fore Probe",marker='+')
    # ax.scatter(Re_ls,GRATES_AFT,label="Aft Probe",marker='x')
    # ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Growth Rate')
    # ax.grid()
    # plt.legend()
    # fig.suptitle("Steady State Supercritical Growth Rate AR(2)")
    # fig.savefig("Steady State Supercritical Growth Rate AR(2)")
    # plt.show()
    
        
        
        