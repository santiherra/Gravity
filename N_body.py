# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import mysympint
import numba

njit = numba.njit 


# FUNCTIONS
@njit
def dSdt(q, p, m): # q and p (N, 2) arrays, m (1, N) arrays
    qdot = np.diag(1/m)@p
    pdot = np.zeros((len(m), 2))
    for i in range(len(m)):
        for j in range(len(m)):
            if j!=i:
                aux = q[i]-q[j]
                d = np.sqrt(np.sum(aux**2))  
                pdot[i] += -m[i]*m[j]*aux/(d**3+1e-8) # 1e-8 term added to avoid dividing by 0
    return qdot, pdot  # qdot and pdot (N, 2) arrays

def pos_mom(nx, ny):
    fx, fy = list(np.linspace(-int(nx/2), int(nx/2), nx)), list(np.linspace(-int(ny/2), int(ny/2), ny))
    q0 = np.array([[x, y] for x in fx for y in fy])*.5  # Generates a lattice
    p0 = np.diag(m)@(np.random.rand(nx*ny, 2)-.5)*.5 
    return q0, p0


# ANIMATION
nx = int(input('# of particles in x dir.: '))
ny = int(input('# of particles in y dir.: '))
N = nx*ny
m = .01*np.ones(N)  # Masses
q0, p0 = pos_mom(nx, ny)
tf, dt = 10, .01  # tf = Final simulation time, dt = simulation step
t = np.linspace(0, tf, int(tf/dt))

q_s, p_s = mysympint.verlet(q0, p0, dSdt, tf, dt, m)

fig, axes = plt.subplots(1, 1)
lines = []
for j in range(N):
    l_aux, = axes.plot([], [], 'o')
    lines = np.concatenate((lines, [l_aux]))
li = float(input('Limit of plot: '))
timer = axes.text(.6*li, .6*li, '', bbox=dict(facecolor='white', edgecolor='black'))
lines = np.concatenate((lines, [timer]))
axes.set_xlim(-li, li)
axes.set_ylim(-li, li)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title(str(len(m)) + ' bodies simulation')

def animate(i):
    for j in range(N):
        lines[j].set_data((q_s[i])[j, 0], (q_s[i])[j, 1])
    lines[-1].set_text("t = " + str(round(t[i], 2)))
    return lines

ani = FuncAnimation(fig, animate, frames=len(t), interval=10, blit=True)
plt.show()