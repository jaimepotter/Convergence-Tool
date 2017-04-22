# -*- coding: utf-8 -*-
"""
Created on Wed Mar  29 16:05:57 2017

@author: Graham Potter
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button


def update(val):
    sigma = setSD.val
    plots.clear()
    del ax.lines[:];
    # sample size for slider 
    N = 100
    # num of trials to average error of SD and mean over
    numTimes = 300
    
    #fig3 = plt.figure(figsize=(14,7),edgecolor = 'black',frameon=False,tight_layout=True)
    #axAll = fig3.add_subplot(111)
    i = 1
    mtot = []
    sdtot = []
    while i < numTimes:
        s = np.random.normal(mu, sigma, N)
        i2 = 1
        m = []
        sd = []
        while i2 < len(s):
            m.append(100*np.abs(np.mean(s[1:i2])-mu)/mu)
            sd.append(100*np.abs(np.std(s[1:i2+1])-sigma)/sigma)
            i2 = i2 + 1
        mtot.append(m)
        sdtot.append(sd)
        #axAll.plot(m,color=[1,0,0,.5])
        #axAll.plot(sd,color=[0,0,1,.5])
        i = i + 1
    
    # collect all values from each sample generated and avg error for every step 1 -> N
    i3 = 1
    temp = []
    temp2 = []
    while i3 < N-1:
        i4 = 1
        sumM = []
        sumSD = []
        while i4 < numTimes-1:
            sumM.append(mtot[i4][i3])
            sumSD.append(sdtot[i4][i3])
            i4 = i4 + 1
        temp.append(np.mean(sumM))
        temp2.append(np.mean(sumSD))
        i3 = i3 + 1
    ax.plot(temp,color='red',linewidth=2)
    ax.plot(temp2,color='blue',linewidth=2)
    ax.relim()
    ax.autoscale_view(True,True,True)
    ax.set_title('comparison for '+str(sigma)+' SD')
    plt.draw()
    plt.show()
    bins=30
    ax4.hist(s, bins, normed=True)
    ax4.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    ax4.relim()
    ax4.autoscale_view(True,True,True)
    ax4.draw()
    ax4.show()
    print('here')
    return mtot

fig5 = plt.figure(figsize=(14,7),edgecolor = 'black',frameon=False,tight_layout=True)
ax4 = plt.subplot(111)
fig2 = plt.figure(figsize=(14,7),edgecolor = 'black',frameon=False,tight_layout=True)
mu, sigma = 1, .68 # mean and standard deviation
s = np.random.normal(mu, sigma, 2000)
mean = []
SD = []
plots = []

i = 1
while i < len(s):
    mean.append(100*np.abs(np.mean(s[1:i])-mu)/mu)
    SD.append(100*np.abs(np.std(s[1:i+1])-sigma)/sigma)
    i = i+1
count, bins, ignored = plt.hist(s, 30, normed=True)
p1 = plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
fig = plt.figure(figsize=(14,7),edgecolor = 'black',frameon=False,tight_layout=False)
ax = plt.axes([0.05,.2,.9,.7]) 
plt.title('comparison for '+str(sigma)+' SD')
plt.ylabel('percent error')
plt.xlabel('sample size N')
mean, = plt.plot(mean,label='mean error',color='red')
SD, = plt.plot(SD,label='SD error',color='blue')
legend = ax.legend(loc='right center', shadow=True)
axSlide = plt.axes([0.1,.05,.8,.05]) 
setSD = Slider(axSlide, 'Set SD', 0.1, 10, valinit=sigma)


setSD.on_changed(update)
plt.show()