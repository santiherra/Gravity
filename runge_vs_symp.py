# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import mysympint
import scienceplots

plt.style.use(['science', 'no-latex'])


# FUNCTIONS
def Hamiltonian(q, p, m):
    x1, y1, x2, y2, x3, y3 = q
    px1, py1, px2, py2, px3, py3 = p
    m1, m2, m3 = m
    K = .5*m1*(px1**2 +py1**2)+.5*m2*(px2**2 +py2**2)+.5*m3*(px3**2 +py3**2)
    U = -m1*m2/np.sqrt((x1-x2)**2+(y1-y2)**2)-m1*m3/np.sqrt((x1-x3)**2+(y1-y3)**2)-m2*m3/np.sqrt((x2-x3)**2+(y2-y3)**2)
    return U + K

def dSdt_s(q, p , m):
    qdot = np.diag(1/m)@p 
    pdot = np.zeros((len(m), 2))
    v12, v13, v23 = q[0]-q[1], q[0]-q[2], q[1]-q[2]
    d12, d13, d23 = np.sqrt(np.sum(v12**2)), np.sqrt(np.sum(v13**2)), np.sqrt(np.sum(v23**2))
    pdot[0] = -m[0]*m[1]*v12/d12**3 - m[0]*m[2]*v13/d13**3
    pdot[1] = m[0]*m[1]*v12/d12**3 - m[1]*m[2]*v23/d23**3
    pdot[2] = m[0]*m[2]*v13/d13**3 + m[2]*m[1]*v23/d23**3
    return qdot, pdot

def dSdt_ode(S, t, m):
    m1, m2, m3 = -m
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = S
    r12, r13, r23 = np.sqrt((x1-x2)**2+(y1-y2)**2)**3, np.sqrt((x1-x3)**2+(y1-y3)**2)**3, np.sqrt((x3-x2)**2+(y3-y2)**2)**3
    return [px1/np.abs(m1), py1/np.abs(m1), px2/np.abs(m2), py2/np.abs(m2), px3/np.abs(m3), py3/np.abs(m3),
            m2*(x1-x2)/r12+m3*(x1-x3)/r13, m2*(y1-y2)/r12+m3*(y1-x3)/r13, m1*(x2-x1)/r12+m3*(x2-x3)/r23, 
            m1*(y2-y1)/r12+m3*(y2-y3)/r23, m1*(x3-x1)/r13+m3*(x3-x2)/r23, m1*(y3-y1)/r13+m3*(y3-y2)/r23]


# ANIMATIONS
m = np.ones(3)
S0 = np.array([1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1])
t = np.linspace(0, 5, 1000)
tf, dt = t[-1], t[1]-t[0]

sol = odeint(dSdt_ode, S0, t, args=(m,))
x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = sol.T
q_ode = np.array([x1, y1, x2, y2, x3, y3])
p_ode = np.array([px1, py1, px2, py2, px3, py3])
H_ode = Hamiltonian(q_ode, p_ode, m)

q0, p0 = np.array([[1, 0], [0, 0], [-1, 0]]), np.array([[0, 1], [0, 0], [0, -1]])
q_s, p_s = mysympint.verlet(q0, p0, dSdt_s, tf, dt, m)
x1_s, y1_s = (q_s[:, 0, :])[:, 0], (q_s[:, 0, :])[:, 1]
x2_s, y2_s = (q_s[:, 1, :])[:, 0], (q_s[:, 1, :])[:, 1]
x3_s, y3_s = (q_s[:, 2, :])[:, 0], (q_s[:, 2, :])[:, 1]
px1_s, py1_s = (p_s[:, 0, :])[:, 0], (p_s[:, 0, :])[:, 1]
px2_s, py2_s = (p_s[:, 1, :])[:, 0], (p_s[:, 1, :])[:, 1]
px3_s, py3_s = (p_s[:, 2, :])[:, 0], (p_s[:, 2, :])[:, 1]
q_H_s = np.array([x1_s, y1_s, x2_s, y2_s, x3_s, y3_s])
p_H_s = np.array([px1_s, py1_s, px2_s, py2_s, px3_s, py3_s])
H_s = Hamiltonian(q_H_s, p_H_s, m)

fig1, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(t, H_s, label='Verlet')
ax.plot(t, H_ode, label='odeint')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Energy comparison')
ax.legend()
ax.grid()

fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
l1ode, = axes[0].plot([], [], 'o', color='darkred')
l2ode, = axes[0].plot([], [], 'o', color='darkgreen')
l3ode, = axes[0].plot([], [], 'o', color='darkblue')
sombra1, = axes[0].plot([], [], color='red', lw=.5)
sombra2, = axes[0].plot([], [], color='green', lw=.5)
sombra3, = axes[0].plot([], [], color='blue', lw=.5)
arrow1 = axes[0].quiver(x1[0], y1[0], px1[0], py1[0], minshaft =1, minlength=0, scale=18, width=.005)
arrow2 = axes[0].quiver(x2[0], y2[0], px2[0], py2[0], minshaft =1, minlength=0, scale=18, width=.005)
arrow3 = axes[0].quiver(x3[0], y3[0], px3[0], py3[0], minshaft =1, minlength=0, scale=18, width=.005)
axes[0].set_xlim(-1.5, 1.5)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Runge kutta')
l1, = axes[1].plot([], [], 'o', color='purple')
l2, = axes[1].plot([], [], 'o', color='gold')
l3, = axes[1].plot([], [], 'o', color='darkolivegreen')
sombra1_s, = axes[1].plot([], [], color='mediumorchid', lw=.5)
sombra3_s, = axes[1].plot([], [], color='olivedrab', lw=.5)
arrow1_s = axes[1].quiver(x1_s[0], y1_s[0], px1_s[0], py1_s[0], minshaft =1, minlength=0, scale=18, width=.005)
arrow3_s = axes[1].quiver(x3_s[0], y3_s[0], px3_s[0], py3_s[0], minshaft =1, minlength=0, scale=18, width=.005)
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(-2, 2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Symplectic')
fig2.suptitle('ORBIT COMPARISON')

def animate(i):
    trail=50
    l1ode.set_data(x1[i], y1[i])
    l2ode.set_data(x2[i], y2[i])
    l3ode.set_data(x3[i], y3[i])
    sombra1.set_data(x1[i-trail:i], y1[i-trail:i])
    sombra2.set_data(x2[i-trail:i], y2[i-trail:i])
    sombra3.set_data(x3[i-trail:i], y3[i-trail:i])
    arrow1.set_offsets(np.array([x1[i], y1[i]]).T)
    arrow1.set_UVC(px1[i], py1[i])
    arrow2.set_offsets(np.array([x2[i], y2[i]]).T)
    arrow2.set_UVC(px2[i], py2[i])
    arrow3.set_offsets(np.array([x3[i], y3[i]]).T)
    arrow3.set_UVC(px3[i], py3[i])
    l1.set_data((q_s[i])[0, 0], (q_s[i])[0, 1])
    l2.set_data((q_s[i])[1, 0], (q_s[i])[1, 1])
    l3.set_data((q_s[i])[2, 0], (q_s[i])[2, 1])
    sombra1_s.set_data(x1_s[i-trail:i], y1_s[i-trail:i])
    sombra3_s.set_data(x3_s[i-trail:i], y3_s[i-trail:i])
    arrow1_s.set_offsets(np.array([x1_s[i], y1_s[i]]).T)
    arrow1_s.set_UVC(px1_s[i], py1_s[i])
    arrow3_s.set_offsets(np.array([x3_s[i], y3_s[i]]).T)
    arrow3_s.set_UVC(px3_s[i], py3_s[i])
    return l1ode, l2ode, l3ode, sombra1, sombra2, sombra3, arrow1, arrow2, arrow3, l1, l2, l3, sombra1_s, sombra3_s, arrow1_s, arrow3_s, 

ani = FuncAnimation(fig2, animate, frames = len(t), interval=10, blit=True)
plt.show()
