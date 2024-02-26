import numpy as np

def verlet(q0, p0, fun, tf, dt, m):
    Nt = int(tf/dt)
    q_sol = [q0]
    p_sol = [p0]
    for i in range(1, Nt+1):
        pdot0 = fun(q0, p0, m)[1]
        pm = p0+dt/2*pdot0
        qdotm = fun(q0, pm, m)[0]
        qN = q0+dt*qdotm
        pdotN = fun(qN, p0, m)[1]
        pN = pm+dt/2*pdotN
        q_sol = np.concatenate((q_sol, [qN]))
        p_sol = np.concatenate((p_sol, [pN]))
        q0, p0 = qN, pN
    return q_sol, p_sol