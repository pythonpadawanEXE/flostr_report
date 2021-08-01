
from pathlib import Path

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import meshio
import os
import cylinder
from matplotlib.ticker import MaxNLocator
import modred as mr
from scipy.sparse import load_npz
from scipy.signal import find_peaks
import skfem
import skfem.io.json
import skfem.visuals.matplotlib as skfemvis
import json

def main(filename: Path,Rey, start_n: int = 10000,end_n: int =15000,r:int =21):
    W = load_npz(Path(str(os.path.dirname(__file__))+'\\'+'SIM_XDMF\\'
                               +"st08_navier_stokes_cylinder"+
                               "mass_Re_{}".format(str(Rey).replace('.','-'))+".npz"))
    W = W.A #to dense matrix
    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()
        ssf = 11
        qT = np.array(
            [
                reader.read_data(k)[1]["vorticity"]           #["velocity"][:, :2].flatten("F")
                for k in range(start_n, end_n,ssf) #change ssf?
                #for k in range(3000,3300)
            ]
        )
        
     
    x = qT.T


    DMD = mr.compute_DMD_arrays_snaps_method(x,inner_product_weights=W,mode_indices=range(r),
                                             max_num_eigvals=r)
    dT = 0.001
    omega = np.log(DMD.eigvals)/dT
    
    '''
    find leading mode determined by the index of the nomalised magnitude
    and print corresponding frequency and growth rate
    '''
    print("Eigen Values",DMD.eigvals)

    
    '''
    Calculate Number of Cycles
    '''
    peaks_A, _A = find_peaks(qT.T[4][1:])
    troughs_A, t_A = find_peaks(-qT.T[4][1:])
    cycles = round((len(peaks_A)+len(troughs_A))/2,0)
    '''
    '''
    f_j = np.angle(DMD.eigvals)/(2*np.pi*dT*ssf)  #eq A9 (Anton Burtsev et. al. 2021 )
    g_j = np.log(np.abs(DMD.eigvals))/(dT*ssf)      #eq A10 (Anton Burtsev et. al. 2021 )
    val = list(zip(g_j,f_j))
    t = len(x[0])                              #time range length
    tsps = t/cycles                             # time steps per cycle
    print("Number of Cycles",cycles,"time steps per cycle",tsps)
    #print( "Re is", Re,val)
    
    
    fig,ax = plt.subplots()
    point = 512
    ax.plot(np.array(range(t))*ssf,x.T[:,point])
    ax.set(xlabel="Time-Step", ylabel="vorticity" ,title='Plot of Vorticity at ({:.3f}, {:.3f})'.format(*points[point, :2]))
    ax.grid()
    #fig.savefig('Plot of Vorticity at ({:.3f}, {:.3f})'.format(*points[point, :2]))
    plt.show()

   # DMD Spectrum Graph
    vortall = np.loadtxt("VORTALL.txt")
    x = vortall
    D = mr.compute_DMD_arrays_snaps_method(x,mode_indices=range(r),
                                             max_num_eigvals=r)
    theta = np.arange(0,100,1)*2*np.pi/100
    fig,ax = plt.subplots()
    ax.plot(np.sin(theta),np.cos(theta),'--k')
    ax.scatter(D.eigvals.real,D.eigvals.imag,marker='o',label="Vortall")
    ax.scatter(DMD.eigvals.real,DMD.eigvals.imag,marker='o',label= "st08-NS")
    ax.set(xlabel = 'Re(λ)', ylabel = 'Im(λ)', title='Eigen Values of %d DMD Modes'%r)
    plt.legend()
    fig.savefig("DMD Spectrum- {} modes - Re {:.2f}.png".format(r,Re))
    plt.show()
    
    
    
    Phi = DMD.exact_modes
    Phi_u, Phi_s, Phi_vh  = np.linalg.svd(Phi, full_matrices=False) #SVD rearrange faster than pinv for larger matrices is lstsq faster?
    Phi_pinv = Phi_vh.T / Phi_s @ Phi_u.T #eq A8 (Anton Burtsev et. al. 2021 )
    x = qT.T
    x1 = x[:,0]
    b  = Phi_pinv@x1                #eq A7 (Anton Burtsev et. al. 2021 )
    
    
    AMPS   = np.abs(b)[::-1]/np.linalg.norm(np.abs(b)[::-1])
    FREQS  = f_j[::-1] #omega.imag?
    GRATES = g_j[::-1] #omega.real?
    MODES  = DMD.exact_modes.T[::-1]
    OMEGA_N = omega[::-1]
    b_N    = b[::-1]
    
    #Remove conjugate negative frequencies
    AMPS   = np.delete(AMPS, np.argwhere((FREQS < 0)))
    GRATES = np.delete(GRATES, np.argwhere((FREQS < 0)))
    MODES = np.delete(MODES, np.argwhere( (FREQS< 0)),0) 
    OMEGAS = np.delete(OMEGA_N, np.argwhere( (FREQS< 0)),0) 
    Bs = np.delete(b_N, np.argwhere( (FREQS< 0)),0) 
    FREQS  = np.delete(FREQS, np.argwhere((FREQS < 0)))
    
   
    time_dynamics = np.zeros((MODES.shape[0],t),dtype=complex)
    for i in range(t):
        time_dynamics[:,i] = (Bs*np.exp(OMEGAS*(i)*dT))
    
    
    #X_dmd = (MODES@time_dynamics).real
    
    fig,ax = plt.subplots()
    ax.set(xlabel = 'Snapshot', ylabel = 'Amplitude', title=str('Frequency Plot at Re={}'.format(Rey)) )
    for i in range(11):
        ax.plot(list(np.array(range(t))*ssf)[:],(time_dynamics[i].real)[:], label="Mode {}- Freq- {}".format(i,round(FREQS[i],2)))
        peaks_A, _A = find_peaks(time_dynamics[i])
        troughs_A, t_A = find_peaks(-time_dynamics[i])
        cycles = round((len(peaks_A)+len(troughs_A))/2,0)
        print("For mode",i,"Freq is",cycles/(end_n-start_n/1000))
    ax.legend()    
    plt.show()
    
    fundamental_arg = np.argmax(FREQS)
    for idx,FREQ_i in enumerate(FREQS):
        if FREQ_i  > 0 and FREQ_i < FREQS[fundamental_arg]: 
            fundamental_arg = idx
    #dt applied at plotting stage instead
    return GRATES[fundamental_arg]*dT,FREQS[fundamental_arg]*dT
   
if __name__ == "__main__":
    from argparse import ArgumentParser
    U =  1.4994965504069229 #np.max(uv0[inlet_dofs]) , can a
   # nu =  (0.1*U)/np.arange(81,86,1)
    nu =  np.unique(((0.1*U)/np.arange(90,151,10)))
    #nu = [0.1*U/118]
    Re_ls = []
    GRATES_DMD_REAL = []
    FREQS_DMD_REAL = []


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
        Rey =Re
        GRATE,FREQ = main(Path(args.filename),Rey)
        
        GRATES_DMD_REAL.append(GRATE)
        FREQS_DMD_REAL.append(FREQ)
        
        if not os.path.exists(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'):
            try:
                os.makedirs(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        
        with open(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'+'\GRATES_DMD.json', 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(GRATES_DMD_REAL, f, indent=2) 
        with open(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'+'\FREQS_DMD.json', 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(FREQS_DMD_REAL, f, indent=2) 
       
            

        with open(str(os.path.dirname(__file__))+'\SS_SCRIT_JSON'+'\Re_ls.json', 'w') as f:
                    # indent=2 is not needed but makes the file human-readable
                    json.dump(Re_ls, f, indent=2) 
        
   