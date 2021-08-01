# -*- coding: utf-8 -*-
import meshio
import os
import numpy as np


x = []
files = [int(f.strip('.vtk')[3:]) for f in os.listdir(r'pd-0.20\VTK') 
           if f.endswith('.vtk')]
files = files[1:]
files.sort()

for file in files:
        m = meshio.read(f'pd-0.20\\VTK\\20_{file}.vtk')
        x.append(m.cell_data["vorticity"][0][:, 0])

x = np.array(x).T