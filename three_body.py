# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation


# FUNCTIONS
def orbitevol (S, t, m):
    m1, m2, m3 = m
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = S
    r12, r13, r23 = np.sqrt((x1-x2)**2+(y1-y2)**2)**3, np.sqrt((x1-x3)**2+(y1-y3)**2)**3, np.sqrt((x3-x2)**2+(y3-y2)**2)**3
    return [vx1, vy1, vx2, vy2, vx3, vy3, m2*(x1-x2)/r12+m3*(x1-x3)/r13, m2*(y1-y2)/r12+m3*(y1-x3)/r13, 
            m1*(x2-x1)/r12+m3*(x2-x3)/r23, m1*(y2-y1)/r12+m3*(y2-y3)/r23, m1*(x3-x1)/r13+m3*(x3-x2)/r23,
            m1*(y3-y1)/r13+m3*(y3-y2)/r23]


# ANIMATION
m = -np.ones(3)
S0 = np.array([1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1])
t = np.linspace(0, 5, 1000)

Sol = odeint(orbitevol, S0, t=t, args=(m,))
x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = Sol.T
#vx1, vy1, vx2, vy2, vx3, vy3 = vx1/np.sqrt(vx1**2+vy1**2), vy1/np.sqrt(vx1**2+vy1**2), vx2/np.sqrt(vx2**2+vy2**2), vy2/np.sqrt(vx2**2+vy2**2), vx3/np.sqrt(vx3**2+vy3**2), vy3/np.sqrt(vx3**2+vy3**2)
#plt.plot([x1[51], x2[51], x3[51], x1[51]], [y1[51], y2[51], y3[51], y1[51]],'-o')

fig= plt.figure()
ln, = plt.plot([], [], '.', color='black', lw=2)
sombra1, = plt.plot([], [], color='red', lw=.5)
sombra2, = plt.plot([], [], color='green', lw=.5)
sombra3, = plt.plot([], [], color='blue', lw=.5)
arrow1 = plt.quiver(x1[0], y1[0], vx1[0], vy1[0], minshaft =1, minlength=0, scale=18, width=.005)
arrow2 = plt.quiver(x2[0], y2[0], vx2[0], vy2[0], minshaft =1, minlength=0, scale=18, width=.005)
arrow3 = plt.quiver(x3[0], y3[0], vx3[0], vy3[0], minshaft =1, minlength=0, scale=18, width=.005)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title(r"THREE BODIES ORBITING$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
n = len(t)

def anime(i):
  trail = 50
  ln.set_data([x1[i], x2[i], x3[i]], [y1[i], y2[i], y3[i]])
  sombra1.set_data(x1[i-trail:i+1], y1[i-trail:i+1])
  sombra2.set_data(x2[i-trail:i+1], y2[i-trail:i+1])
  sombra3.set_data(x3[i-trail:i+1], y3[i-trail:i+1])
  arrow1.set_offsets(np.array([x1[i], y1[i]]).T)
  arrow1.set_UVC(vx1[i], vy1[i])
  arrow2.set_offsets(np.array([x2[i], y2[i]]).T)
  arrow2.set_UVC(vx2[i], vy2[i])
  arrow3.set_offsets(np.array([x3[i], y3[i]]).T)
  arrow3.set_UVC(vx3[i], vy3[i])
  return ln, sombra1, sombra2, sombra3, arrow1, arrow2, arrow3,

ani = FuncAnimation(fig, anime, frames=len(t), interval=10, blit=True)
plt.show()