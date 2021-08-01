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

def main(filename: Path,Rey, n: int = 3000,r:int =21):
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
                for k in range(-n, 0,ssf) #change ssf?
                #for k in range(3000,3300)
            ]
        )
        
     
    x = qT.T
    X1 = x[:int(np.floor(x.shape[0]/2)),:]
    X2 = x[int(np.ceil(x.shape[0]/2)):,:]
    
    #DMD = mr.compute_DMD_arrays_direct_method(X1,X2,mode_indices=range(r),max_num_eigvals=r)
    DMD = mr.compute_DMD_arrays_snaps_method(x,inner_product_weights=W,mode_indices=range(r),
                                             max_num_eigvals=r)
    dT = 0.001#*ssf
    omega = np.log(DMD.eigvals)/dT
    
    '''
    find leading mode determined by the index of the nomalised magnitude
    and print corresponding frequency and growth rate
    '''
    print("Eigen Values",DMD.eigvals)
    #print("lead mode maybe:")
    #max_idx = list(DMD.spectral_coeffs).index(max(DMD.spectral_coeffs)) 
    
    #print(omega.real[max_idx],omega.imag[max_idx])
    
    
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
    for i in range(MODES.shape[0]):
        ax.plot(list(np.array(range(t))*ssf)[:],(time_dynamics[i].real)[:], label="Mode {}- Freq- {}".format(i,round(FREQS[i],2)))
        peaks_A, _A = find_peaks(time_dynamics[i])
        troughs_A, t_A = find_peaks(-time_dynamics[i])
        cycles = round((len(peaks_A)+len(troughs_A))/2,0)
        print("For mode",i,"Freq is",cycles/(n/1000))
    ax.legend()    
    plt.show()
    
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
    
   
    try:
        mesh = skfem.io.json.from_file("cylinder.json")
    except:
        from cylinder import mesh
        
    element = {"u": skfem.ElementVectorH1(skfem.ElementTriP2()), "p": skfem.ElementTriP1()}
    basis = {
        **{v: skfem.InteriorBasis(mesh, e, intorder=4) for v, e in element.items()},
        "inlet": skfem.FacetBasis(mesh, element["u"], facets=mesh.boundaries["inlet"]),
    }
    
    if not os.path.exists(str(os.path.dirname(__file__))+'\DMD_MODES'):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\DMD_MODES')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(str(os.path.dirname(__file__))+'\DMD_MODES\Re_{}'.format(Re)):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\DMD_MODES\Re_{}'.format(Re))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(str(os.path.dirname(__file__))+'\DMD_MODES\Re_{}\Interval_IV'.format(Re)):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\DMD_MODES\Re_{}\Interval_IV'.format(Re))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
  
    for i,(freq,growth,MODE) in enumerate(zip(FREQS,GRATES, MODES)):
              print("i is",i)
              
              # Real Mode
              
              fig,ax = skfemvis.plt.subplots(figsize=(11,2)) 
              # ax.figure(figsize=(5.5,1))
              skfemvis.plot(basis['p'], MODE.real,ax=ax,vmin = MODES.real.min() , vmax = MODES.real.max(),cmap='jet',shading='gouraud')
              #maybe make basis an axes then plot MODE using imshow?
              #ax.set_title(f"Mode {i:.0f} at Re {Re:.0f}")
              ax.tick_params(
                    bottom=True,
                    left = True)
              ax.margins(x=0,y=0)
              xticks=np.arange(0,23)/10
              yticks=list(np.linspace(0,0.4,5))
              cc=plt.Circle(( 0.2 ,0.2),radius=0.05,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
              ax.add_artist( cc )
              ax.set(aspect=1,xlabel="X/D",
                              ylabel="Y/D",xticks=xticks,yticks=yticks,yticklabels=[i for i in range(-2,3,1)],xticklabels=[i for i in range(-2,21,1)])#,xticks=[i for i in range(-1,11,1)],
                              #yticks=[i for i in range(-2,3,2)])
                                            
              ax.figure.savefig(f"{os.path.dirname(__file__)}"+"\DMD_MODES\Re_{}\Interval_IV".format(Re)
                                      +f"\SSF{ssf}-last{n}-snapshots-grate_{growth}-freq_{freq}-Real-Mode-Re-{Re:.0f}-Mode-{i:.0f}.png",bbox_inches = 'tight')  
             
             # Imaginary Mode
            
              fig,ax = skfemvis.plt.subplots(figsize=(11,2)) 
              # ax.figure(figsize=(5.5,1))
              skfemvis.plot(basis['p'], MODE.imag,ax=ax,vmin = MODES.imag.min() , vmax = MODES.imag.max(),cmap='jet',shading='gouraud')
              #maybe make basis an axes then plot MODE using imshow?
              #ax.set_title(f"Mode {i:.0f} at Re {Re:.0f}")
              ax.tick_params(
                    bottom=True,
                    left = True)
              ax.margins(x=0,y=0)
              xticks=np.arange(0,23)/10
              yticks=list(np.linspace(0,0.4,5))
              cc=plt.Circle(( 0.2 ,0.2),radius=0.05,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
              ax.add_artist( cc )
              ax.set(aspect=1,xlabel="X/D",
                              ylabel="Y/D",xticks=xticks,yticks=yticks,yticklabels=[i for i in range(-2,3,1)],xticklabels=[i for i in range(-2,21,1)])#,xticks=[i for i in range(-1,11,1)],
                              #yticks=[i for i in range(-2,3,2)])
                                            
              ax.figure.savefig(f"{os.path.dirname(__file__)}"+"\DMD_MODES\Re_{}\Interval_IV".format(Re)
                                      +f"\SSF{ssf}-last{n}-snapshots-grate_{growth}-freq_{freq}-Imaginary-Mode-Re-{Re:.0f}-Mode-{i:.0f}.png",bbox_inches = 'tight')  
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
        
   
