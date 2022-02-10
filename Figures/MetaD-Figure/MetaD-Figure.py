#! /usr/bin/env python

# Author: Omar Valsson
# Script to generate Metadynamics figure (Figure 3 in version 1.0).
# The figure is for a model potential given by
# F(z)=a*x**4-4*a*x**2+b*x (see parameters below)
# The hill file is taken from a metadynamics run on this 
# model system.

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


gamma=4.0
beta=0.5
a=2.0
b=0.1918
FES_filename="FES.data"
GridMin=-4.0
GridMax=+4.0
GridBins=4000
HillsFile="metad-bf4-s020_hills.data"


x = np.linspace(GridMin,GridMax,GridBins)

FES=a*x**4-4*a*x**2+b*x
FES=FES-FES.min()
FES_gamma=FES/gamma
FES_gamma=FES_gamma-FES_gamma.min()
PDF=np.exp(-beta*FES)
PDF=PDF/np.trapz(y=PDF,x=x)
norm=np.trapz(y=PDF,x=x) 
PDF_gamma=np.exp(-beta*FES_gamma)
PDF_gamma=PDF_gamma/np.trapz(y=PDF_gamma,x=x)
norm_gamma=np.trapz(y=PDF_gamma,x=x) 

DataOut = [x, FES, PDF, FES_gamma, PDF_gamma]
Header="! FIELDS x FES PDF FES_gamma PDF_gamma\n"
Header+="! SET FES(x)=a*x^4-4*a*x^2+b*x\n".format(gamma)
Header+="! SET a={}\n".format(a)
Header+="! SET b={}\n".format(b)
Header+="! SET beta={}\n".format(beta)
Header+="! SET gamma={}".format(gamma)
np.savetxt(FES_filename, np.column_stack(DataOut), header=Header)


Offset=+12.0
xlimits = [-2.5,+2.5]
FES_gamma_offset=FES_gamma+Offset
lw=2
alpha=0.6
fontsize=16
color_normal='#8c564b'
color_gamma='#1f77b4'

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(2,2,figsize=[16.0,8.0],sharex='col')

ax1 = ax[1,1]
ax2 = ax[0,1]
ax3 = ax[1,0]
ax4 = ax[0,0]


ax1.fill_between(x,PDF,alpha=alpha,color=color_normal)
ax1.fill_between(x,PDF_gamma,alpha=alpha,color=color_gamma)
# ax1.plot(x,PDF,linewidth=lw,color=color_normal)
# ax1.plot(x,PDF_gamma,linewidth=lw,color=color_gamma)
ax1.set_xlim(xlimits)
ax1.set_ylim(bottom=0.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("Collective Variable",fontsize=fontsize)
ax1.set_ylabel("Probability",fontsize=fontsize)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.text(-1.25,PDF[np.abs(x - -1.25).argmin()]+0.1,"$\\rho(\mathbf{z})$",color=color_normal,fontsize=fontsize)
ax1.text(-1.00,PDF_gamma[np.abs(x - -1.00).argmin()]+0.05,"$\\tilde{\\rho}(\mathbf{z})\propto [\\rho(\mathbf{z})]^{1/\gamma}$",color=color_gamma,fontsize=fontsize)


NumGaussians = [20000, 1200, 416, 50]

# ax3.set_xticks([])
ax3.set_xticks([0, 400, 800, 1200],labels=["0", "400", "800", "1200"],fontsize=fontsize)
ax3.set_yticks([])
ax3.set_xlim([0,1200])
ax3.set_xlabel("Number of Gaussians Deposited",fontsize=fontsize)
ax3.set_ylabel("Gaussian Height",fontsize=fontsize)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ax4.set_xticks([])
ax4.set_xticks([0, 400, 800, 1200],labels=["0", "400", "800", "1200"],fontsize=fontsize)
ax4.set_yticks([])
ax4.set_xlim([0,1200])
# ax4.set_xlabel("Number of Gaussians Deposited",fontsize=fontsize)
ax4.set_ylabel("Collective Variable",fontsize=fontsize)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

hillsdata=np.loadtxt(HillsFile)
time=hillsdata[:,0]
pos=hillsdata[:,1]
biasf=hillsdata[0,4]
height=(1.0-(1.0/biasf))*hillsdata[:,3]
sigma=hillsdata[0,2]
# color_gaussian='#d62728'
color_gaussian=color_gamma
color_gaussian='blue'
for k in NumGaussians:
    BIAS=np.zeros(x.size)
    for i in range(k):
        BIAS += height[i]*np.exp(-0.5*((x-pos[i])/sigma)**2)
    ax2.fill_between(x,FES,BIAS+FES,where=BIAS>=0.0,alpha=alpha)
    ax2.plot(x,BIAS+FES,linewidth=lw,label="{}".format(k))
    ax3.plot(time[:k],height[:k])
    ax4.plot(time[:k],pos[:k])
ax2.plot(x,FES, linewidth=lw,color=color_normal)
ax2.legend(loc=[0.0,0.65],
           fontsize=12,
           title="Number of Gaussian")
#ax2.plot(x[FES_gamma_offset >= FES],
#         FES_gamma_offset[FES_gamma_offset >= FES],
#         linewidth=lw,
#         color=color_gamma)
#ax2.fill_between(x,FES,FES_gamma_offset,where=FES_gamma_offset >= FES,alpha=alpha,color=color_gamma)
ax2.set_xlim(xlimits)
ax2.set_ylim([0.0,+26])
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.set_xlabel("Collective Variable",fontsize=fontsize)
ax2.set_ylabel("Free Energy",fontsize=fontsize)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
# ax2.text(-2.2,FES[np.abs(x - -2.2).argmin()]+2.0,"$A(\mathbf{z})$",color=color_normal,fontsize=fontsize)
# ax2.text(0.2,FES[np.abs(x - 0.2).argmin()]+BIAS[np.abs(x - 0.2).argmin()]+2.0,"$A(\mathbf{z})+U_{\mathrm{bias}}(\mathbf{z})$",color=color_gamma,fontsize=fontsize)


timestampStr = datetime.now().strftime("%d-%b-%Y_%H%M")
plt.savefig("MetaD-Figure.png")
# plt.savefig("MetaD-Figure.{}.png".format(timestampStr))
plt.show()




