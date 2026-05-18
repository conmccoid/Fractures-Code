from petsc4py import PETSc
import numpy as np
from Plotters import plotEnergyLandscape2D, plotInterpolantAcc

def cbt(fp,x,p,res, tol1=1e-16, tol2=1e-4):
    """
    Perform cubic backtracking line search.

    Parameters:
    - fp: fixed point function for AltMin
    - x: current solution vector
    - p: Newton search direction vector
    - res: AltMin step
    """

    E0 = fp.updateEnergies(x)[2]  # initial energy
    # if fp.rank==0:
    #     print(f"Initial Energy: {E0}")
    alpha = 1.0  # initial step length
    fp.updateGradF(x)
    gp = p.dot(fp.gradF)

    # initial energies for AltMin and MSPIN
    xcopy=x.copy()
    xcopy.axpy(alpha,p)
    E1 = fp.updateEnergies(xcopy)[2]  # new energy
    # Efp= fp.updateEnergies(x + res)[2] # energy after AltMin step
    # print(f"Energy at Newton step: {E1}, Energy at AltMin step: {Efp}")

    # cubic backtracking
    first_time=True
    while (E1 > E0 - alpha*tol2*gp) and (alpha > tol1):
        if first_time==True:
            alpha_0 = 1.0
            E_prev = E1
            alpha = -gp/(2*(E1 - E0 - gp))
            first_time=False
        else:
            array1=np.array([[1/alpha**2, -1/alpha_0**2],[-alpha_0/alpha**2, alpha/alpha_0**2]])
            array2=np.array([E1 - E0 - gp * alpha, E_prev - E0 - gp * alpha_0])
            array_out = array1.dot(array2) / (alpha - alpha_0)
            a=array_out[0]
            b=array_out[1]
            if a == 0:
                alpha_new = -gp/(2*b)
            else:
                alpha_new = (-b + np.sqrt(b**2 - 3*a*gp)) / (3*a)
            alpha_0 = alpha
            E_prev = E1
            alpha = alpha_new
            if alpha > 0.5 * alpha_0: # not sure if this is necessary
                alpha = 0.5 * alpha_0
            # print(f"Backtracking step length: {alpha}, Energy: {E1}, Target energy: {E0}")
        xcopy.waxpy(alpha,p,x) # replacing xcopy=x and then xcopy.axpy(alpha,p) to avoid creating multiple copies, may not work
        E1 = fp.updateEnergies(xcopy)[2]
    #     print(f"Backtracking step length: {alpha}, Energy: {E1}, Target energy: {E0 - alpha*tol2*gp}")
    # print(f"Final step length: {alpha}")
    p.scale(alpha)
    xcopy.destroy()

def pbt(fp,x,q,p):
    """
    Construct quadratic 2D polynomial that interpolates the residual over a parallelogram
    
    Parameters:
    - fp: function handling example
    - x: current solution
    - q: direction towards the AltMin step
    - p: direction towards the Newton step
    """
    fp.updateGradF(x)
    e=p.dot(fp.gradF)
    d=q.dot(fp.gradF)
    
    xq=x.copy()
    xq.axpy(1,q)
    xp=x.copy()
    xp.axpy(1,p)
    xpq=x.copy()
    xpq.axpy(1,q)
    xpq.axpy(1,p)
    qp=q.copy()
    qp.axpy(1,p)

    E0=fp.updateEnergies(x)[2]
    Eq=fp.updateEnergies(xq)[2]
    Ep=fp.updateEnergies(xp)[2]
    Epq=fp.updateEnergies(xpq)[2]
    f=E0
    c=Ep-e-f
    a=Eq-d-f
    b=Epq + f - Ep - Eq

    r=4*a*c-b**2
    alpha = (-2*c*d + b*e)/r # step in q (AltMin)
    beta = (-2*a*e + b*d)/r # step in p (MSPIN)

    coeffs = [a,b,c,d,e,f, r, alpha, beta]
    E_list = [E0, Eq, Ep, Epq]

    xp.destroy()
    xq.destroy()
    xpq.destroy()

    return coeffs, E_list, qp

def pm1(fp, x, q, p):
    """
    Find the minimum of a quadratic 2D polynomial that interpolates the residual in a parallelogram.
    The minimum is restricted to the original parallelogram.

    Parameters:
    - fp: function handling example
    - x: current solution
    - q: direction towards the AltMin step
    - p: direction towards the Newton step

    nb: some kinks to work out, not clear what to do when r==0
    """
    # need to do DB trick first - does gradF need to have a commensurate sign change?
    [a,b,c,d,e,f,r,alpha_opt,beta_opt], [E0,Eq,Ep,Epq], qp = pbt(fp,x,q,p)
    angle = np.arccos(np.clip(q.dot(p)/(q.norm()*p.norm()), -1, 1)) # angle between AltMin and Newton steps

    E_list=[Eq, Ep, Epq]
    v_list=[q.copy(), p.copy(), qp]
    beta_list=[0, 1, 1]
    alpha_list=[1, 0, 1]
    step_list=["AltMin", "Newton", "Both"]

    set_list=[[alpha_opt, beta_opt, "Parallelogram interior"]]
    if a!=0:
        set_list.append([-d/(2*a),0,"AltMin boundary"])
        set_list.append([(-d - b)/(2*a),0,"Newton + AltMin boundary"])
    if c!=0:
        set_list.append([-e/(2*c),1,"Newton boundary"])
        set_list.append([(-e - b)/(2*c),1,"AltMin + Newton boundary"])
    for [alpha, beta, step] in set_list:
        if (alpha>=0) & (alpha<=1) & (beta>=0) & (beta<=1):
            v=q.copy()
            v.scale(alpha)
            v.axpy(beta,p)
            xv=x.copy()
            xv.axpy(1,v)
            Ev=fp.updateEnergies(xv)[2]
            xv.destroy()

            v_list.append(v)
            E_list.append(Ev)
            alpha_list.append(alpha)
            beta_list.append(beta)
            step_list.append(step)

    min_index=np.argmin(E_list)
    result=v_list[min_index].copy()
    alpha=alpha_list[min_index]
    beta=beta_list[min_index]
    # print(f"Chosen step: {step_list[min_index]}, Energy: {E_list[min_index]}")

    # clean-up
    for v in v_list:
        v.destroy()
    qp.destroy()

    return result, angle, alpha, beta, alpha_opt, beta_opt, r, a

def pm2(fp, x, q, p, filename=None):
    """
    Find the minimum of a quadratic 2D polynomial that interpolates the residual in a parallelogram.
    There is no restriction on the minimum.
    
    Parameters:
    - fp: function handling example
    - x: current solution
    - q: direction towards the AltMin step
    - p: direction towards the Newton step
    """
    [a,b,c,d,e,f,r,alpha,beta], _, qp = pbt(fp,x,q,p)
    angle = np.arccos(np.clip(q.dot(p)/(q.norm()*p.norm()), -1, 1)) # angle between AltMin and Newton steps
    result=q.copy()
    result.scale(alpha)
    result.axpy(beta,p)
    qp.destroy()
    if filename is not None and r<=0:
        plotInterpolantAcc(fp,x,q,p,[a,b,c,d,e,f], filename=filename)
    return result, angle, alpha, beta, alpha, beta, r, a

def pm3(fp, x, q, p, filename=None):
    """
    Find the minimum of a quadratic 2D polynomial that interpolates the residual in a parallelogram.
    
    Parameters:
    - fp: function handling example
    - x: current solution
    - q: direction towards the AltMin step
    - p: direction towards the Newton step
    """
    [a,b,c,d,e,f,r,alpha,beta], [E0,Eq,Ep,Epq], qp = pbt(fp,x,q,p)
    qp.destroy()

    if r>0:
        if a>0:
            result=q.copy()
            result.scale(alpha)
            result.axpy(beta,p)
        else:
            # plotEnergyLandscape2D(fp,x,q,p,[beta,alpha],filename=filename)
            # while Eq < E0:
            #     E0=Eq
            #     q.scale(2)
            #     xq=x.copy()
            #     xq.axpy(1,q)
            #     Eq=fp.updateEnergies(xq)[2]
            # q.scale(0.5)
            # result=q.copy()
            result=q.copy()
            result.scale(alpha)
            result.axpy(beta,p)
    else:
        # alpha = -b
        # beta = (a-c) - np.sqrt((a-c)**2 + b**2)
        plotEnergyLandscape2D(fp,x,q,p,[beta, alpha],filename=filename, coeffs=[a,b,c,d,e,f])
        result=q.copy()
        result.scale(alpha)
        result.axpy(beta,p)

    return result