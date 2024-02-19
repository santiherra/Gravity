# IMPORT PACKAGES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.optimize import root
plt.style.use(['dark_background'])


# ORBITAL FUNCTIONS
def a(aph, e): # semi-major axis
    return aph/(1+e)

def orbit(e, a, phi, the): # rotation and translation elliptical parametrization
    x = (a*np.cos(phi)-a*e)
    y = a*np.sqrt(1-e**2)*np.sin(phi)*np.cos(the)
    z = -a*np.sqrt(1-e**2)*np.sin(phi)*np.sin(the)
    return x, y, z

phi = np.linspace(0, 2*np.pi, 500)

def kepler(t, E, T0, e): # function for the Newton-Raphson Method (target: eccentric anomaly)
    M = 2*np.pi/T0*t
    k = M-E+e*np.sin(E)
    return k


#PLANETARY DATA
e_r = np.array([.206, .007, .017, .094])
aph_r = np.array([69.8, 108.9, 152.1, 249.3])
a_r = a(aph_r, e_r)
T0_r = 24*3600*np.array([88, 108.9, 365.2, 249.3])
the_r = np.pi/180*np.array([-7, -3.4, 0, -1.8])
x_me, y_me, z_me = orbit(e_r[0], a_r[0], phi, the_r[0])
x_v, y_v, z_v = orbit(e_r[1], a_r[1], phi, the_r[1])
x_e, y_e, z_e = orbit(e_r[2], a_r[2], phi, the_r[2])
x_ma, y_ma, z_ma = orbit(e_r[3], a_r[3], phi, the_r[3])

e_g = np.array([.049, .052, .047, .01])
aph_g = np.array([816.4, 1506.5, 3001.4, 4558.9])
a_g = a(aph_g, e_g)
T0_g = 24*3600*np.array([4331, 10747, -30589, 59800])
the_g = np.pi/180*np.array([-1.3, -2.5, -.7, -1.8])
x_j, y_j, z_j = orbit(e_g[0], a_g[0], phi, the_g[0])
x_s, y_s, z_s = orbit(e_g[1], a_g[1], phi, the_g[1])
x_u, y_u, z_u = orbit(e_g[2], a_g[2], phi, the_g[2])
x_n, y_n, z_n = orbit(e_g[3], a_g[3], phi, the_g[3])


# PLANETARY PLOTS

fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
axes[0].plot(0, 0, 0, '*', label='Sun',color='yellow')
axes[0].plot(x_me, y_me, z_me, label='Mercury')
axes[0].plot(x_v, y_v, z_v, label='Venus')
axes[0].plot(x_e, y_e, z_e, label='Earth')
axes[0].plot(x_ma, y_ma, z_ma, label='Mars')
line1_r, = axes[0].plot([], [], [], 'o')
line2_r, = axes[0].plot([], [], [], 'o')
line3_r, = axes[0].plot([], [], [], 'o')
line4_r, = axes[0].plot([], [], [], 'o')
axes[0].set_zlim(-max(x_ma), max(x_ma))
axes[0].set_xlabel('X axis')
axes[0].set_ylabel('Y axis')
axes[0].set_zlabel('Z axis')
axes[0].set_title('Rocky Planets Orbits', fontweight='bold')
axes[0].legend()
axes[1].plot(0, 0, 0, '*', label='Sun',color='yellow')
axes[1].plot(x_j, y_j, z_j, label='Jupiter')
axes[1].plot(x_s, y_s, z_s, label='Saturn')
axes[1].plot(x_u, y_u, z_u, label='Uranus')
axes[1].plot(x_n, y_n, z_n, label='Neptune')
line1_g, = axes[1].plot([], [], [], 'o')
line2_g, = axes[1].plot([], [], [], 'o')
line3_g, = axes[1].plot([], [], [], 'o')
line4_g, = axes[1].plot([], [], [], 'o')
axes[1].set_zlim(-max(x_u), max(x_u))
axes[1].set_xlabel('X axis')
axes[1].set_ylabel('Y axis')
axes[1].set_zlabel('Z axis')
axes[1].set_title('Giant Planets Orbits', fontweight='bold')
axes[1].legend()


# ANIMATIONS

def animate(i):
    E0 = np.zeros(4)
    x_rocky = np.ones(4)
    y_rocky = np.ones(4)
    z_rocky = np.ones(4)
    for j in range(0, len(T0_r)):
        k = lambda E: kepler(i*24*3600, E, T0_r[j], e_r[j])
        #dkdE = lambda E: -1+e_r[j]
        E0[j] = root(k, E0[j]).x
        x_rocky[j], y_rocky[j], z_rocky[j] = orbit(e_r[j], a_r[j], E0[j], the_r[j])
    line1_r.set_data([x_rocky[0]], [y_rocky[0]])
    line1_r.set_3d_properties([z_rocky[0]])
    line2_r.set_data([x_rocky[1]], [y_rocky[1]])
    line2_r.set_3d_properties([z_rocky[1]])
    line3_r.set_data([x_rocky[2]], [y_rocky[2]])
    line3_r.set_3d_properties([z_rocky[2]])
    line4_r.set_data([x_rocky[3]], [y_rocky[3]])
    line4_r.set_3d_properties([z_rocky[3]])
    E0 = np.zeros(4)
    x_giant = np.ones(4)
    y_giant = np.ones(4)
    z_giant = np.ones(4)
    for w in range(0, len(T0_g)):
        k = lambda E: kepler(i*24*3600*100, E, T0_g[w], e_g[w])
        #dkdE = lambda E: -1+e_g[j]
        E0[w] = root(k, E0[w]).x
        x_giant[w], y_giant[w], z_giant[w] = orbit(e_g[w], a_g[w], E0[w], the_g[w])
    line1_g.set_data([x_giant[0]], [y_giant[0]])
    line1_g.set_3d_properties([z_giant[0]])
    line2_g.set_data([x_giant[1]], [y_giant[1]])
    line2_g.set_3d_properties([z_giant[1]])
    line3_g.set_data([x_giant[2]], [y_giant[2]])
    line3_g.set_3d_properties([z_giant[2]])
    line4_g.set_data([x_giant[3]], [y_giant[3]])
    line4_g.set_3d_properties([z_giant[3]])
    return line1_r, line2_r, line3_r, line4_r, line1_g, line2_g, line3_g, line4_g,

anime = FuncAnimation(fig, animate, frames=300, interval=10)
plt.show()
#anime_rocky.save('RockyPlanets.gif', writer='pillow', fps=10, dpi=100)
