# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:56:24 2021

@author: jakee
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
import modred as mr
from scipy.sparse import load_npz

def main(filename: Path,Rey, n: int = 3000,r:int =21):
    frame = 0
    vortall = np.loadtxt("VORTALL.txt")
    
    fig,ax = plt.subplots()
    x_len= vortall[:, frame].reshape(449, 199).shape[0]
    y_len= vortall[:, frame].reshape(449, 199).shape[1]
    x_axis = np.arange(0,(x_len+int(np.ceil(x_len/9))),int(np.ceil(x_len/9)))
    x_axis_norm = ((x_axis-50)/50).astype(int)
    y_axis = np.arange(0,(y_len+int(np.ceil(y_len/4))),int(np.ceil(y_len/4)))
    y_axis_norm = ((y_axis-100)/50).astype(int)[::-1]
    
    image = imshow(vortall[:, frame].reshape(449, 199).T,vmin = vortall.min(),vmax=vortall.max(),cmap='jet')
    cc=plt.Circle(( 49 , 99 ),radius=25,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
    ax.add_artist( cc ) 
    ax.set_xticks(x_axis)
    ax.set_yticks(y_axis)
               
    ax.set_xticklabels(x_axis_norm)
    ax.set_yticklabels(y_axis_norm)
    ax.set(xlabel="X/D",ylabel="Y/D")
    ax.get_figure().savefig(f"{Path(__file__).stem}-{frame}.png")
    
   
    x = vortall
    Y = np.hstack((x,x))
    nx = 199
    ny = 449
    for i in range(x.shape[1]):
         xflip = (np.flipud(x[:,i].reshape(nx,ny,order='F'))).reshape(nx*ny,order='F')
         Y[:,i+x.shape[1]] = -xflip;
         
    vortavg = np.mean(Y,1)
    X = Y - vortavg[np.newaxis].T@np.ones((1,Y.shape[1]))
    POD = mr.compute_POD_arrays_direct_method(X,mode_indices=range(r))
    MODES = POD.modes                                        
    eiv = POD.eigvals[:r]**0.5
   
    
   
    
    for i,(eig,MODE) in enumerate(zip(eiv, MODES.T)):
               print("i is",i)
               x_len= MODE.reshape(449, 199).shape[0]
               y_len= MODE.reshape(449, 199).shape[1]
               
               x_axis = np.arange(0,(x_len+int(np.ceil(x_len/9))),int(np.ceil(x_len/9)))
               x_axis_norm = ((x_axis-50)/50).astype(int)
               y_axis = np.arange(0,(y_len+int(np.ceil(y_len/4))),int(np.ceil(y_len/4)))
               y_axis_norm = ((y_axis-100)/50).astype(int)[::-1] 
               
               
               fig, ax = plt.subplots(figsize=(9,2)) 
               
               image =plt.imshow(MODE.reshape(449, 199).T,vmin = -0.02,vmax =0.02,cmap='jet') #vmin = -0.02,vmax =0.02,
               # plt.colorbar()
               cc=plt.Circle(( 49 , 99 ),radius=25,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
               ax.add_artist( cc ) 
               ax.set_xticks(x_axis)
               ax.set_yticks(y_axis)
               
               ax.set_xticklabels(x_axis_norm)
               ax.set_yticklabels(y_axis_norm)
               ax.set(xlabel="X/D",ylabel="Y/D")
              # plt.colorbar()
              # ax.set_title(f"Mode (real) @ Amp = {AMP:.2f}, FREQ = {FREQ:.2f}")
               ax.get_figure().savefig(f"{os.path.dirname(__file__)}\POD_TEST_MODES\\"
                                       +f"{Path(__file__).stem}-Real-Mode-Re-{Rey:.2f}-Mode-{i:.2f}.png",bbox_inches = 'tight')  
    '''
    Plot Modal Energies %
    '''
    modes = np.arange(1,(r+1),1)
    modal_energy = eiv*100/np.sum(eiv)
    fig,ax = plt.subplots()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.plot(modes,modal_energy,marker=',')
    ax.set(xlabel = 'Mode Number', ylabel = 'Modal Energy %', title=str('Modal Energy Distribution at Re={}'.format(Rey)) )
    # ax.grid()
    ax.scatter(modes[:-1],modal_energy[:-1] , marker=r'$\bigcirc$')
    plt.xticks(modes[:-1], modes[:-1])
    fig.savefig("Modal_Energies_{}.png".format(Rey))
    
    plt.show()
    
    '''
    Plot Modal Energies % Cumulative
    '''
    modes = np.arange(1,(r+1),1)
    modal_energy = eiv*100/np.sum(eiv)
    fig,ax = plt.subplots()
    
    modal_energy_cumulative = []
    for i in range(len(modal_energy)):
        modal_energy_cumulative.append(sum(modal_energy[:i+1]))
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.plot(modes,modal_energy,marker=',')
    ax.set(xlabel = 'Mode Number', ylabel = 'Modal Energy % Cumulative', title=str('Modal Energy Distribution at Re={}'.format(Rey)) )
    # ax.grid()
    ax.scatter(modes[:-1],modal_energy_cumulative[:-1],marker=r'$\bigcirc$')
    plt.xticks(modes[:-1], modes[:-1])
    fig.savefig("Modal_Energies_{}_Cumulative.png".format(Rey))
    
    plt.show()
   
    '''
    Plot V eigen vectors versus time?
    '''
   
    time_ = np.arange(1,x.shape[1]*2+1)
    fig,ax = plt.subplots()
    ax.set(xlabel = 'Time', ylabel = 'Amplitude', title=str('Time Coefficient Vectors for Vh at Re={}'.format(Rey)) )
    for i in range(3):
        ax.plot(time_,POD.eigvecs[:,i], label="Mode {}".format(i+1))
        
    ax.legend()    
    plt.show()
   
   
  
    
if __name__ == "__main__":
    
    from argparse import ArgumentParser
    
    nu = list(0.1/np.arange(100,101,2))
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
        
   
