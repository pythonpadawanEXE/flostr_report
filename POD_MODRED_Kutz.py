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
import cylinder
# from matplotlib.ticker import MaxNLocator
import skfem
import skfem.io.json
import skfem.visuals.matplotlib as skfemvis

def main(filename: Path,Rey, n: int = 3000,r:int =21):
    W = load_npz(Path(str(os.path.dirname(__file__))+'\\'+'SIM_XDMF\\'
                               +"st08_navier_stokes_cylinder"+
                               "mass_Re_{}".format(str(Rey).replace('.','-'))+".npz"))
    W = W.A
    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()
        ssf = 1
        qT = np.array(
            [
                reader.read_data(k)[1]["vorticity"]
                for k in range(-n, 0,ssf)
            ]
        )
        
     
    xT = qT - qT.mean(0)  
    x  = xT.T
    
    
    
    POD = mr.compute_POD_arrays_direct_method(x,inner_product_weights=W,mode_indices=range(r))
    MODES = POD.modes                                        
    eiv = POD.eigvals[:r]**0.5
    
    
    
    vortall = np.loadtxt("VORTALL.txt")
    x = vortall
    Y = np.hstack((x,x))
    nx = 199
    ny = 449
    for i in range(x.shape[1]):
         xflip = (np.flipud(x[:,i].reshape(nx,ny,order='F'))).reshape(nx*ny,order='F')
         Y[:,i+x.shape[1]] = -xflip;
         
    vortavg = np.mean(Y,1)
    X = Y - vortavg[np.newaxis].T@np.ones((1,Y.shape[1]))
    POD_vortall = mr.compute_POD_arrays_direct_method(X,mode_indices=range(r))
    MODES_vortall = POD_vortall.modes                                        
    eiv_vortall = POD_vortall.eigvals[:r]**0.5
    '''
    Plot Modal Energies %
    '''
    modes = np.arange(1,(r+1),1)
    modal_energy = eiv*100/np.sum(eiv)
    modal_energy_vortall = eiv_vortall*100/np.sum(eiv_vortall)
    fig,ax = plt.subplots()
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.plot(modes,modal_energy,marker=',')
    ax.set(xlabel = 'Mode Number', ylabel = 'Modal Energy %', title=str('Modal Energy Distribution at Re={}'.format(Rey)) )
    # ax.grid()
    ax.scatter(modes[:-1],modal_energy_vortall[:-1] , marker=r'$\bigcirc$',label ="Vortall")
    ax.scatter(modes[:-1],modal_energy[:-1] , marker=r'X',label ="st08-NS")
    plt.xticks(modes[:-1], modes[:-1])
    ax.legend()
    fig.savefig("Modal_Energies_{}.png".format(Rey))
    
    dif = np.max(np.array(modal_energy_vortall)-np.array(modal_energy))
    print("max energy difference is",dif)
    plt.show()
    
    '''
    Plot Modal Energies % Cumulative
    '''
    modes = np.arange(1,(r+1),1)
    modal_energy = eiv*100/np.sum(eiv)
    fig,ax = plt.subplots()
    
    modal_energy_cumulative = []
    modal_energy_cumulative_vortall = []
    for i in range(len(modal_energy)):
        modal_energy_cumulative.append(sum(modal_energy[:i+1]))
        modal_energy_cumulative_vortall.append(sum(modal_energy_vortall[:i+1]))
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.plot(modes,modal_energy,marker=',')
    ax.set(xlabel = 'Mode Number', ylabel = 'Modal Energy % Cumulative', title=str('Modal Energy Distribution at Re={}'.format(Rey)) )
    # ax.grid()
    ax.scatter(modes[:-1],modal_energy_cumulative_vortall[:-1] , marker=r'$\bigcirc$',label ="Vortall")
    ax.scatter(modes[:-1],modal_energy_cumulative[:-1],marker=r'X',label ="st08-NS")
    ax.legend()
    plt.xticks(modes[:-1], modes[:-1])
    fig.savefig("Modal_Energies_{}_Cumulative.png".format(Rey))
    
    plt.show()
   
    '''
    Plot V eigen vectors versus time?
    '''
   
    # time_ = np.arange(1,n+1,ssf)
    # fig,ax = plt.subplots()
    # ax.set(xlabel = 'Time', ylabel = 'Amplitude', title=str('Time Coefficient Vectors for Vh at Re={}'.format(Rey)) )
    # for i in range(3):
    #     ax.plot(time_,POD.eigvecs[:,i], label="Mode {}".format(i+1))
        
    # ax.legend()    
    # plt.show()
    # with meshio.xdmf.TimeSeriesWriter(Path(str(os.path.dirname(__file__))+'\\'+'POD_XDMF\\'
    #                            +os.path.splitext(str(os.path.basename(__file__)))[0]+'_Modes_{}'.format(r)+
    #                            "_Re_{}.xdmf".format(str(round(Rey,2)).replace('.','-')))) as writer:
    
    #     writer.write_points_cells(points, cells)
    #     print(points.shape)
    #     for f, mod in zip(range(r),MODES.T): #using lamda.real breaks program?
    #                 writer.write_data(f, point_data={"vorticity ": mod})
    
    try:
        mesh = skfem.io.json.from_file("cylinder.json")
    except:
        from cylinder import mesh
    
    element = {"u": skfem.ElementVectorH1(skfem.ElementTriP2()), "p": skfem.ElementTriP1()}
    basis = {
        **{v: skfem.InteriorBasis(mesh, e, intorder=4) for v, e in element.items()},
        "inlet": skfem.FacetBasis(mesh, element["u"], facets=mesh.boundaries["inlet"]),
    }
    
    if not os.path.exists(str(os.path.dirname(__file__))+'\POD_MODES'):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\POD_MODES')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(str(os.path.dirname(__file__))+'\POD_MODES\Re_{}'.format(Re)):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\POD_MODES\Re_{}'.format(Re))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(str(os.path.dirname(__file__))+'\POD_MODES\Re_{}\Interval_IV'.format(Re)):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\POD_MODES\Re_{}\Interval_IV'.format(Re))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
  
    for i,(eig,MODE) in enumerate(zip(eiv, MODES.T)):
              print("i is",i)
              fig,ax = skfemvis.plt.subplots(figsize=(11,2)) 
              # ax.figure(figsize=(5.5,1))
              skfemvis.plot(basis['p'], MODE,ax=ax,vmin = MODES.min() , vmax = MODES.max(),cmap='jet',shading='gouraud')
              #maybe make basis an axes then plot MODE using imshow?
              #ax.set_title(f"Mode {i:.0f} at Re {Re:.0f}")
              ax.tick_params(
                    bottom=True,
                    left = True)
              ax.margins(x=0,y=0)
              xticks=list(np.linspace(0,2.2,12))
              yticks=list(np.linspace(0,0.4,5))
              cc=plt.Circle(( 0.2 ,0.2),radius=0.05,fill=True ,facecolor ='grey', edgecolor='black',linewidth=1.2)
              ax.add_artist( cc )
              ax.set(aspect=1,xlabel="X/D",
                              ylabel="Y/D",xticks=xticks,yticks=yticks,yticklabels=[i for i in range(-2,3,1)],xticklabels=[i for i in range(-1,11,1)])#,xticks=[i for i in range(-1,11,1)],
                              #yticks=[i for i in range(-2,3,2)])
                                            
              ax.figure.savefig(f"{os.path.dirname(__file__)}"+"\POD_MODES\Re_{}\Interval_IV".format(Re)
                                      +f"\SSF{ssf}-last{n}-snapshots-Real-Mode-Re-{Re:.0f}-Mode-{i:.0f}.png",bbox_inches = 'tight')  
  
    
if __name__ == "__main__":
    
    from argparse import ArgumentParser
    U =  1.4994965504069229
    nu = (0.1*U)/np.arange(150,151,10)
    Re_ls =[]
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
    
        main(Path(args.filename),round(Re,2))
        
        
   
