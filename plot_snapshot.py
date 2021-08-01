# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 18:03:17 2021

@author: jakee
"""

"""
@author: jakee
"""
from pathlib import Path

import numpy as np
# import dask.array as da
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
import meshio
import os
import cylinder
# from matplotlib.ticker import MaxNLocator
import modred as mr
from scipy.sparse import load_npz
import skfem
import skfem.io.json
import skfem.visuals.matplotlib as skfemvis
#from matplotlib.pyplot import subplots

def main(filename: Path,Re,start_n: int = 0000,end_n:int = 60000,r:int = 21):
    W = load_npz(Path(str(os.path.dirname(__file__))+'\\'+'SIM_XDMF\\'
                               +"st08_navier_stokes_cylinder"+
                               "mass_Re_{}".format(str(Re).replace('.','-'))+".npz"))
    W = W.A
    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()
        ssf = 500
        qT = np.array(
            [
                reader.read_data(k)[1]["vorticity"]
                for k in range(start_n,end_n,ssf)
            ]
        )
        
     
    xT = qT - qT.mean(0)  
    x  = xT.T
    
                    
    '''
    Plot POD snapshots with matplotlib and skfem
    '''
    try:
        mesh = skfem.io.json.from_file("cylinder.json")
    except:
        from cylinder import mesh
    
    element = {"u": skfem.ElementVectorH1(skfem.ElementTriP2()), "p": skfem.ElementTriP1()}
    basis = {
        **{v: skfem.InteriorBasis(mesh, e, intorder=4) for v, e in element.items()},
        "inlet": skfem.FacetBasis(mesh, element["u"], facets=mesh.boundaries["inlet"]),
    }
    
    if not os.path.exists(str(os.path.dirname(__file__))+'\POD_SNAPSHOTS'):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\POD_SNAPSHOTS')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(str(os.path.dirname(__file__))+'\POD_SNAPSHOTS\Re_{}'.format(Re)):
        try:
            os.makedirs(str(os.path.dirname(__file__))+'\POD_SNAPSHOTS\Re_{}'.format(Re))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
  
    for i,(num,snapshot) in enumerate(zip(np.arange(x.shape[1]), x.T)):
              print("i is",i)
              fig,ax = skfemvis.plt.subplots(figsize=(11,2))
              # ax.figure(figsize=(5.5,1))
              skfemvis.plot(basis['p'], snapshot,ax=ax,cmap='jet',shading='gouraud')
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
              ax.set(xlabel="X/D",
                              ylabel="Y/D",xticks=xticks,yticks=yticks,yticklabels=[i for i in range(-2,3,1)],xticklabels=[i for i in range(-2,21,1)])#,xticks=[i for i in range(-1,11,1)],
                              #yticks=[i for i in range(-2,3,2)])

              
                                                             
              ax.figure.savefig(f"{os.path.dirname(__file__)}"+"\POD_SNAPSHOTS\Re_{}\Snapshot_{}.png".format(Re,start_n+num*ssf),bbox_inches = 'tight')                                                            
    


   
if __name__ == "__main__":
 
    from argparse import ArgumentParser
    
    U =  1.4994965504069229 #np.max(uv0[inlet_dofs])
    # nu =  np.unique(np.concatenate(((0.1*U)/np.arange(70,71,10),(0.1*U)/np.arange(70,81,1))))
    # nu =  np.unique(((0.1*U)/np.arange(90,151,10)))
    nu =  np.unique(((0.1*U)/np.arange(40,41,10)))
    # nu = [0.1*U/78]
    amp_eig = []
    amp_eig_sqrt = []
    Re_ls =[]
    #U = 1
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