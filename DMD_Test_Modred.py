# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:54:51 2021

@author: jakee

modified verson of pod.py
"""

from pathlib import Path

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import meshio
import os
import cylinder
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm as CM
import modred as mr
from scipy.signal import find_peaks

def get_continuous_cmap(rgb_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in rgb_list.
        If float_list is provided, each color in rgb_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        rgb_list: list of rgb colours
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def main(filename: Path,Rey, n: int = 5000,r:int =21):

    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()

        qT = np.array(
            [
                reader.read_data(k)[1]["velocity"][:, :2].flatten("F")
                for k in range(-n, 0)
            ]
        )
        
        
    frame = 0
    vortall = np.loadtxt("VORTALL.txt")
    ax = imshow(vortall[:, frame].reshape(449, 199).T)
    ax.get_figure().savefig(f"{Path(__file__).stem}-{frame}.png")
    
    
    
    
    x = vortall
   
    DMD = mr.compute_DMD_arrays_direct_method(x,mode_indices=range(r),max_num_eigvals=r)
    dT = 0.02
    omega = np.log(DMD.eigvals)/dT
    x1 = x[:,0]    
    b = np.linalg.lstsq(DMD.exact_modes,x1,rcond=None)[0]
    f_j = np.angle(DMD.eigvals)/(2*np.pi*dT)  #eq A9 (Anton Burtsev et. al. 2021 )
    g_j = np.log(np.abs(DMD.eigvals))/dT      #eq A10 (Anton Burtsev et. al. 2021 )
    '''
   
    DMD Spectrum Graph
    
    '''
    theta = np.arange(0,100,1)*2*np.pi/100
    fig,ax = plt.subplots()
    ax.plot(np.sin(theta),np.cos(theta),'--k')
    ax.scatter(DMD.eigvals.real,DMD.eigvals.imag,marker='o')
    ax.set(xlabel = 'Re(λ)', ylabel = 'Im(λ)', title='Eigen Values of %d DMD Modes'%r)
    #ax.grid()
    fig.savefig("DMD Spectrum")
    plt.show()
    
    
    '''
    '''
   
   
    AMPS   = np.abs(b)[::-1]/np.linalg.norm(np.abs(b)[::-1])
    FREQS  = f_j[::-1]
    GRATES = g_j[::-1]
    MODES  = DMD.exact_modes.T[::-1]
    OMEGA_N = omega[::-1]
    b_N    = b[::-1]
    
    AMPS   = np.delete(AMPS, np.argwhere((FREQS < 0)))
    GRATES = np.delete(GRATES, np.argwhere((FREQS < 0)))
    MODES = np.delete(MODES, np.argwhere( (FREQS< 0)),0) 
    OMEGAS = np.delete(OMEGA_N, np.argwhere( (FREQS< 0)),0) 
    Bs = np.delete(b_N, np.argwhere( (FREQS< 0)),0) 
    FREQS  = np.delete(FREQS, np.argwhere((FREQS < 0)))
    
    n= vortall.shape[1]
    time_dynamics = np.zeros((MODES.shape[0],n),dtype=complex)
    t = len(x[0])
    ssf = 1
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
        print("For mode",i,"Freq is",cycles/(n/1000))
    ax.legend()    
    plt.show()
    
     #Plot DMD Modes for Example
   
    if not os.path.exists(str(os.path.dirname(__file__))+'\DMD_TEST_MODES'):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\DMD_TEST_MODES')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            
    CC = np.loadtxt("CCmap.txt")
    CC = list(map(tuple, CC))
    CC = get_continuous_cmap(CC)
    CC = 'jet'
    
    
    for i,(FREQ,GRATE,MODE) in enumerate(zip(FREQS, GRATES, MODES)):
            print("i is",i)
            x_len= MODE.reshape(449, 199).shape[0]
            y_len= MODE.reshape(449, 199).shape[1]
            
            x_axis = np.arange(0,(x_len+int(np.ceil(x_len/9))),int(np.ceil(x_len/9)))
            x_axis_norm = ((x_axis-50)/50).astype(int)
            y_axis = np.arange(0,(y_len+int(np.ceil(y_len/4))),int(np.ceil(y_len/4)))
            y_axis_norm = ((y_axis-100)/50).astype(int)[::-1]
            
            
            fig, ax = plt.subplots(figsize=(9,2)) 
            
            image =plt.imshow(MODE.real.reshape(449, 199).T,vmin = -0.02,vmax =0.02,cmap=CC)
            cc=plt.Circle(( 49 , 99 ),radius=25,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
            ax.add_artist( cc ) 
            ax.set_xticks(x_axis)
            ax.set_yticks(y_axis)
            
            ax.set_xticklabels(x_axis_norm)
            ax.set_yticklabels(y_axis_norm)
            ax.set(xlabel="X/D",ylabel="Y/D")
            # plt.colorbar()
           # ax.set_title(f"Mode (real) @ Amp = {AMP:.2f}, FREQ = {FREQ:.2f}")
            ax.get_figure().savefig(f"{os.path.dirname(__file__)}\DMD_TEST_MODES\\"
                                    +f"{Path(__file__).stem}-GRATE-{GRATE}-Freq-{FREQ}-Real-Mode.png")
            
            fig, ax = plt.subplots(figsize=(9,2))
            
            image =plt.imshow(MODE.imag.reshape(449, 199).T,vmin = -0.02,vmax =0.02,cmap=CC)
            cc=plt.Circle(( 49 , 99 ),radius=25,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
            ax.add_artist( cc )
           
            ax.set_xticks(x_axis)
            ax.set_yticks(y_axis)
            
            ax.set_xticklabels(x_axis_norm)
            ax.set_yticklabels(y_axis_norm)
            ax.set(xlabel="X/D",ylabel="Y/D")
            # plt.colorbar()
            #ax.set_title(f"Mode (imaginary) @ Amp = {AMP:.2f}, FREQ = {FREQ:.2f}")
            ax.get_figure().savefig(f"{os.path.dirname(__file__)}\DMD_TEST_MODES\\"
                                    +f"{Path(__file__).stem}-GRATE-{GRATE}-Freq-{FREQ}-Imag-Mode.png")
    
    
    fig,ax = plt.subplots(1,2,sharey='row')
    ax[0].set_yscale('log')
    ax[0].stem(FREQS,AMPS,linefmt='C5' ,markerfmt='C5^')
    ax[0].set(xlabel = 'Frequency', ylabel = 'Amplitude')
    ax[1].stem(GRATES,AMPS,linefmt='C5' ,markerfmt='C5^')
    ax[1].set(xlabel = 'Growth Rate')
    fig.savefig("DMD Spectra")
    plt.show()
    
    fig,ax = plt.subplots(2)
    ax[0].set_yscale('log')
    ax[0].stem(FREQS,AMPS,linefmt='C5' ,markerfmt='C5^')
    ax[0].set(xlabel = 'Frequency', ylabel = 'Amplitude')
    ax[1].set_yscale('log')
    ax[1].stem(GRATES,AMPS,linefmt='C5' ,markerfmt='C5^')
    ax[1].set(xlabel = 'Growth Rate')
    fig.savefig("DMD Spectra V2")
    plt.show()
    

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    nu = list(np.arange(0.001,0.0012,0.0002)) #list of kinematic viscosity
    amp_eig = []
    Re_ls =[]
    U = 1
    from cylinder import radius as rad
    Diam = 2*rad
    
    
   
    for i in range(len(nu)):
        Re = round((U*Diam)/nu[i],4)
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
    
        amp_eig.append(main(Path(args.filename),round(Re,2)))
        Re_ls.append(Re)
        
    