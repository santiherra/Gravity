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
                pdot[i] += -m[i]*m[j]*aux/(d**3+1e-6) # 1e-6 term added to avoid dividing by 0
    return qdot, pdot  # qdot and pdot (N, 2) arrays

def request(method):
    if method == 'polar':
        N = int(input('Number of particles: '))
        M = float(input('Mean mass: '))
        s = float(input('Mass std: '))
        m = np.random.normal(M, s, N)
        R = float(input('Radius of space distribution: '))
        th, r = 2*np.pi*np.random.rand(N), R*np.random.rand(N)
        x, y = r*np.cos(th), r*np.sin(th)
        q0 = np.array([x, y]).T
        P = float(input('Maximum momentum: '))
        p = P*np.random.rand(N)
        px, py = -p*np.sin(th), p*np.cos(th)
        p0 = np.array([px, py]).T
        tf = float(input('Simulation time: '))
        dt = float(input('Time step: '))
        return m, q0, p0, tf, dt
    elif method == 'lattice':
        nx = int(input('Number of particles in x dir.: '))
        ny = int(input('Number of particles in y dir.: '))
        N = nx*ny
        M = float(input('Mean mass: '))
        s = float(input('Mass std: '))
        m = np.random.normal(M, s, N)
        fx, fy = list(np.linspace(-int(nx/2), int(nx/2), nx)), list(np.linspace(-int(ny/2), int(ny/2), ny))
        Q = float(input('Lattice semi-amplitude: '))
        P = float(input('Maximum momentum: '))
        q0 = np.array([[x, y] for x in fx for y in fy])*Q  
        p0 = np.diag(m)@(np.random.rand(nx*ny, 2)-.5)*P
        tf = float(input('Simulation time: '))
        dt = float(input('Time step: '))
        return m, q0, p0, tf, dt
    elif method == 'random':
        N = int(input('Number of particles: '))
        M = float(input('Mean mass: '))
        s = float(input('Mass std: '))
        m = np.random.normal(M, s, N)
        Q = float(input('Space distribution amplitude: '))
        P = float(input('Momentum distribution amplitude: '))
        q0, p0 = Q*np.random.rand(N, 2), P*np.random.rand(N, 2)
        tf = float(input('Simulation time: '))
        dt = float(input('Time step: '))
        return m, q0, p0, tf, dt
    else:
        method = input("Invalid method, try one of the following 'polar', 'lattice', 'random'")
        request(method)


# ANIMATION
''' Suggestions
                M = .001, s =M/10 or M/100 ; tf/dt = 1000
                Polar:  R = 5 for N ~ 100 , P = .0005
                Lattice: Q = .5 , P = .005
'''

method = input('Method for parameter setting: ')
m, q0, p0, tf, dt = request(method)
t = np.linspace(0, tf, int(tf/dt))

q_s, p_s = mysympint.verlet(q0, p0, dSdt, tf, dt, m)

fig, axes = plt.subplots(1, 1)
lines = []
for j in range(len(m)):
    l_aux, = axes.plot([], [], '.')
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
    for j in range(len(m)):
        lines[j].set_data((q_s[i])[j, 0], (q_s[i])[j, 1])
    lines[-1].set_text("t = " + str(round(t[i], 2)))
    return lines

ani = FuncAnimation(fig, animate, frames=len(t), interval=1, blit=True)
plt.show()