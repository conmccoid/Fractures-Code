import numpy as np
import matplotlib.pyplot as plt

def spanAxes(ax):
    ax.axvspan(0,0.275,color='gray',alpha=0.3)
    ax.axvspan(0.275,0.285,color='red',alpha=0.3)
    ax.axvspan(0.285,0.425,color='blue',alpha=0.3)
    ax.axvspan(0.425,0.5,color='green',alpha=0.3)

marker= ['o', 's', 'D', '^', 'v']
color= ['g', 'r', 'c', 'm', 'y']

en_altmin = np.loadtxt(f"output/TBL_CTFM_AltMin.csv", delimiter=',',skiprows=1)
en_mspin = np.loadtxt(f"output/TBL_CTFM_CubicBacktracking.csv",delimiter=',',skiprows=1)
en_para = np.loadtxt(f"output/TBL_CTFM_Parallelogram.csv",delimiter=',',skiprows=1)
en_tri = np.loadtxt(f"output/TBL_CTFM_Triangle.csv",delimiter=',',skiprows=1)
en_tet = np.loadtxt(f"output/TBL_CTFM_Tetrahedron.csv",delimiter=',',skiprows=1)

# energy comparison between AltMin and MSPIN
fig, ax = plt.subplots(3,1)

ax[0].plot(en_altmin[0:51,0],en_altmin[0:51,1],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[0].plot(en_mspin[0:51,0],en_mspin[0:51,1],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)
ax[1].plot(en_altmin[0:51,0],en_altmin[0:51,2],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[1].plot(en_mspin[0:51,0],en_mspin[0:51,2],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)
ax[2].plot(en_altmin[0:51,0],en_altmin[0:51,3],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[2].plot(en_mspin[0:51,0],en_mspin[0:51,3],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)

ax[2].set_xlabel('t')
ax[0].set_ylabel('Elastic energy')
ax[1].set_ylabel('Dissipated energy')
ax[2].set_ylabel('Total energy')
for i in range(3):
    spanAxes(ax[i])
ax[2].legend()

fig.savefig('FIG_CTFM_energy2.pdf')

# energy comparison between AltMin and other methods
fig, ax = plt.subplots(3,1)

# ax[0].plot(en_altmin[0:51,0],en_altmin[0:51,1],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[0].plot(en_para[0:51,0],np.abs(en_para[0:51,1]-en_altmin[0:51,1]),label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax[0].plot(en_tri[0:51,0],np.abs(en_tri[0:51,1]-en_altmin[0:51,1]),label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax[0].plot(en_tet[0:51,0],np.abs(en_tet[0:51,1]-en_altmin[0:51,1]),label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)
# ax[1].plot(en_altmin[0:51,0],en_altmin[0:51,2],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[1].plot(en_para[0:51,0],np.abs(en_para[0:51,2]-en_altmin[0:51,2]),label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax[1].plot(en_tri[0:51,0],np.abs(en_tri[0:51,2]-en_altmin[0:51,2]),label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax[1].plot(en_tet[0:51,0],np.abs(en_tet[0:51,2]-en_altmin[0:51,2]),label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)
# ax[2].plot(en_altmin[0:51,0],en_altmin[0:51,3],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax[2].plot(en_para[0:51,0],np.abs(en_para[0:51,3]-en_altmin[0:51,3]),label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax[2].plot(en_tri[0:51,0],np.abs(en_tri[0:51,3]-en_altmin[0:51,3]),label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax[2].plot(en_tet[0:51,0],np.abs(en_tet[0:51,3]-en_altmin[0:51,3]),label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')
ax[2].set_xlabel('t')
ax[0].set_ylabel('Elastic energy')
ax[1].set_ylabel('Dissipated energy')
ax[2].set_ylabel('Total energy')
for i in range(3):
    spanAxes(ax[i])
ax[2].legend()

fig.savefig('FIG_CTFM_energy3.pdf')

# outer iterations
fig, ax = plt.subplots(1,1)

ax.plot(en_altmin[0:51,0],en_altmin[0:51,5],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax.plot(en_mspin[0:51,0],en_mspin[0:51,5],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)
ax.plot(en_para[0:51,0],en_para[0:51,5],label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax.plot(en_tri[0:51,0],en_tri[0:51,5],label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax.plot(en_tet[0:51,0],en_tet[0:51,5],label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax.set_yscale('log')
ax.set_xlabel('t')
ax.set_ylabel('Outer iterations')
spanAxes(ax)
ax.legend()

fig.savefig('FIG_CTFM_outerIts2.pdf')

# inner iterations
fig, ax = plt.subplots(1,1)

ax.plot(en_mspin[0:51,0],en_mspin[0:51,6],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)
ax.plot(en_para[0:51,0],en_para[0:51,6],label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax.plot(en_tri[0:51,0],en_tri[0:51,6],label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax.plot(en_tet[0:51,0],en_tet[0:51,6],label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax.set_yscale('log')
ax.set_xlabel('t')
ax.set_ylabel('Inner iterations')
spanAxes(ax)
ax.legend()

fig.savefig('FIG_CTFM_innerIts2.pdf')

# average inner iterations per outer iteration for interpolant methods
fig, ax = plt.subplots(1,1)

ax.plot(en_para[0:51,0],en_para[0:51,6]/np.maximum(1,en_para[0:51,5]),label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax.plot(en_tri[0:51,0],en_tri[0:51,6]/np.maximum(1,en_tri[0:51,5]),label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax.plot(en_tet[0:51,0],en_tet[0:51,6]/np.maximum(1,en_tet[0:51,5]),label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax.set_xlabel('t')
ax.set_ylabel('Ave. inner / outer')
spanAxes(ax)
ax.legend()

fig.savefig('FIG_CTFM_innerIts3.pdf')

# wall clock time
fig, ax = plt.subplots(1,1)

ax.plot(en_altmin[0:51,0],en_altmin[0:51,4],label='AltMin',ls='--',marker=marker[0],color=color[0],ms=5)
ax.plot(en_mspin[0:51,0],en_mspin[0:51,4],label='MSPIN',ls='--',marker=marker[1],color=color[1],ms=5)
ax.plot(en_para[0:51,0],en_para[0:51,4],label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax.plot(en_tri[0:51,0],en_tri[0:51,4],label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax.plot(en_tet[0:51,0],en_tet[0:51,4],label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax.set_yscale('log')
ax.set_xlabel('t')
ax.set_ylabel('Time elapsed (s) per load step')
spanAxes(ax)
ax.legend()

fig.savefig('FIG_CTFM_time2.pdf')

# outer iterations comparison between AltMin and interpolant methods
fig, ax = plt.subplots(1,1)

ax.plot(en_para[0:51,0],en_para[0:51,5]/np.maximum(1,en_altmin[0:51,5]),label='Parallelogram',ls='--',marker=marker[2],color=color[2],ms=5)
ax.plot(en_tri[0:51,0],en_tri[0:51,5]/np.maximum(1,en_altmin[0:51,5]),label='Triangle',ls='--',marker=marker[3],color=color[3],ms=5)
ax.plot(en_tet[0:51,0],en_tet[0:51,5]/np.maximum(1,en_altmin[0:51,5]),label='Tetrahedron',ls='--',marker=marker[4],color=color[4],ms=5)

ax.set_yscale('log')
ax.set_xlabel('t')
ax.set_ylabel('Percent AltMin outer iterations')
spanAxes(ax)
ax.legend()

fig.savefig('FIG_CTFM_outerIts3.pdf')