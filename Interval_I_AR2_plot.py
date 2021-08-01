# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:39:05 2021

@author: jakee
"""

import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import os
from cylinder import radius as rad
os.chdir(str(os.path.dirname(__file__))+'\INTERVAL_I_SUBCRIT_JSON')

V = 1.4994965504069229/2  #https://en.wikipedia.org/wiki/Vortex_shedding

D = 2*rad
dt = 0.001
V = V*dt 


Re_ls = [i for i in range(80,69,-1)]

with open("FREQS_FORE.json", 'r') as f:
    FREQS_FORE = [i*D/V for i in json.load(f)]
with open("FREQS_AFT.json", 'r') as f:
    FREQS_AFT = [i*D/V for i in json.load(f)]
with open("FREQS_CHRON1.json", 'r') as f:
    FREQS_CHRON1 = [i*D/V for i in json.load(f)]
with open("FREQS_CHRON1.json", 'r') as f:
    FREQS_CHRON2 = [i*D/V for i in json.load(f)]
with open("FREQS_DMD.json", 'r') as f:
    FREQS_DMD = [i*D/V for i in json.load(f)]
    
with open("GRATES_FORE.json", 'r') as f:
    GRATES_FORE = [i*1/dt for i in json.load(f)]
with open("GRATES_AFT.json", 'r') as f:
    GRATES_AFT = [i*1/dt for i in json.load(f)]
with open("GRATES_CHRON1.json", 'r') as f:
    GRATES_CHRON1 = [i*1/dt for i in json.load(f)]
with open("GRATES_CHRON2.json", 'r') as f:
    GRATES_CHRON2 = [i*1/dt for i in json.load(f)]
with open("GRATES_DMD.json", 'r') as f:
    GRATES_DMD = [i*1/dt for i in json.load(f)]



fig,ax = plt.subplots()
ax.scatter(Re_ls,FREQS_FORE,label="Fore Probe AR(2)",marker='+')
ax.scatter(Re_ls,FREQS_AFT,label="Aft Probe AR(2)",marker='x')
ax.scatter(Re_ls,FREQS_CHRON1,label="Chronos 1 AR(2)",marker=r'$\bigcirc$')
ax.scatter(Re_ls,FREQS_CHRON2,label="Chronos 2 AR(2)",marker='^')
ax.scatter(Re_ls,FREQS_DMD,label="DMD Fundamental",marker='d')
ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Strohaul Number')
ax.grid()
plt.legend()
fig.suptitle("Interval I Frequency") #Time Window 3000-15000 in caption
fig.savefig("Interval I Frequency AR(2)")
plt.show()    
    
    
    
fig,ax = plt.subplots()
ax.scatter(Re_ls,GRATES_FORE,label="Fore Probe AR(2)",marker='+')
ax.scatter(Re_ls,GRATES_AFT,label="Aft Probe AR(2)",marker='x')
ax.scatter(Re_ls,GRATES_CHRON1,label="Chronos 1 AR(2)",marker=r'$\bigcirc$')
ax.scatter(Re_ls,GRATES_CHRON2,label="Chronos 2 AR(2)",marker='^')
ax.scatter(Re_ls,GRATES_DMD,label="DMD Fundamental",marker='d')
ax.set(xlabel = 'Reynolds Number (Re)', ylabel = 'Growth Rate')
ax.grid()
plt.legend()
fig.suptitle("Interval I Growth Rate")
fig.savefig("Interval I Growth Rate AR(2)")
plt.show()