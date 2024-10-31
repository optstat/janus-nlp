import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

def burgers_dyns(x : torch.Tensor, u : torch.Tensor,  nu : float, dx : float) -> torch.Tensor:
    # Compute the spatial gradients
    print(f'x: {x}')
    N = len(u)
    du_dx = np.zeros(N)
    for i in range(0, N-1):
        if i == 0:
            um1 = u[N-1]
        else:
            um1 = u[i-1]
        if i == N-1:
            up1 = u[0]
        else:
            up1 = u[i+1]
        uc = u[i]
        du_dx[i] = -uc*(up1-um1)/(2*dx) + nu*(up1-2*uc+um1)/dx/dx
    return du_dx

def burgers_jac(x : torch.Tensor, u : torch.Tensor,  nu : float, dx : float) -> torch.Tensor:
    # Compute the spatial gradients
    N = len(u)
    jac = np.zeros((N,N))
    for i in range(0, N-1):
        if i == 0:
            um1 = u[N-1]
        else:
            um1 = u[i-1]
        if i == N-1:
            up1 = u[0]
        else:
            up1 = u[i+1]
        uc = u[i]
        # -uc*(up1-um1)/(2*dx) + nu*(up1-2*uc+um1)/dx/dx
        jac[i,i] = (up1-um1)/(2*dx)-2*nu/dx/dx

        if i != 0:
            jac[i,i-1] = uc/(2*dx)+nu/dx/dx
        if i != N-1:
            jac[i,i+1] = -uc/(2*dx)+nu/dx/dx
    return jac

def forward(u : torch.Tensor, x : torch.Tensor, nu : float, dx : float, ft: float) -> torch.Tensor:
  t_span = [0, ft]
  sol = solve_ivp(burgers_dyns, t_span, u, method='Radau',jac=burgers_jac, rtol=1e-3, atol=1e-6, args=(nu, dx))
  return sol.y


if __name__ == "__main__":
    nu = 0.01
    dx = 0.001
    N = int(1.0/dx)
    x = np.linspace(0, 1, N)
    u = np.zeros(N)
    print(f'x: {x}')
    for i in range(N):
      if i < N/2:
        u[i]= -1.0
      else:
        u[i] = 1.0    
    print(f'u: {u}')
    ft = 100.0
    sol = forward(u, x, nu, dx, ft)
    #plot the solution as a 2D plot
    plt.figure()
    plt.plot(x,sol)
    plt.savefig('./images/burgers.png')
    plt.close()
    plt.figure()
    plt.plot(sol[:,-1])
    plt.savefig('./images/burgers_final.png')
    plt.close()
