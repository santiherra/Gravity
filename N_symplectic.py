# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import mysympint


# FUNCTIONS
def Hamiltonian(q, p, m):
    K = p**2/(2*m)
    U = 0
    for i in range(len(m)):
        for j in range(len(m)):
            aux = q[i]-q[j]
            d = np.sqrt(np.sum(aux**2))
            U += -m[i]*m[j]/d
    return K+U

def dSdt(q, p, m) -> np.array: # q and p (N, 2) arrays, m (1, N) array
    qdot = np.diag(1/m)@p
    pdot = np.zeros((len(m), 2))
    for i in range(len(m)):
        for j in range(len(m)):
            if j!=i:
                aux = q[i]-q[j]
                d = np.sqrt(np.sum(aux**2))
                pdot[i] += -m[i]*m[j]*aux/d**3
    return qdot, pdot


# ANIMATION
m = np.ones(3)
q0, p0 = np.array([[1, 0], [0, 0], [-1, 0]]), np.array([[-0.5, 1], [0, 0], [0.5, -1]])
tf, dt = 10, .01 
t = np.linspace(0, tf, int(tf/dt))

q_s, p_s = mysympint.verlet(q0, p0, dSdt, tf, dt, m)

fig, axes = plt.subplots(1, 1)
lines = []
for j in range(len(m)):
    l_aux, = axes.plot([], [], 'o')
    lines = np.concatenate((lines, [l_aux]))
axes.set_xlim(-1.5, 1.5)
axes.set_ylim(-1.5, 1.5)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title(str(len(m)) + ' bodies simulation')

def animate(i):
    for j in range(len(m)):
        lines[j].set_data((q_s[i])[j, 0], (q_s[i])[j, 1])
    return lines

ani = FuncAnimation(fig, animate, frames=len(t), interval=40, blit=True)
plt.show()