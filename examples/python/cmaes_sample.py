import cma
from cma.constraints_handler import AugmentedLagrangian
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.metrics import mean_squared_error


dimension=1000
equality_constraints = [True for _ in range(dimension)]  # Both constraints are inequalities
dx = float(1.0/dimension)
nu = 0.1
ft = 1.0
x = np.linspace(0, 1, dimension)
ui = np.sin(np.linspace(0, 2*np.pi, dimension))

# Initialize the augmented Lagrangian with the correct dimension
aug_lag = AugmentedLagrangian(dimension, equality_constraints)

def dyns_burgers(x : np.ndarray, u  : np.ndarray,  nu : float, dx : float) -> np.ndarray:
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



def jac_burgers(x : np.ndarray, u : np.ndarray,  nu : float, dx : float) -> np.ndarray:
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

def forward(u : np.ndarray, x : np.ndarray, nu : float, dx : float, ft: float) -> np.ndarray:
  t_span = [0, ft]
  sol = solve_ivp(dyns_burgers, t_span, u, method='Radau',jac=jac_burgers, rtol=1e-3, atol=1e-6, args=(nu, dx))
  return sol.y

sol_ref = forward(ui, x, nu, dx, ft)


# Define the objective function
def objective(u):
    x = np.linspace(0, 1, dimension)
    sol=forward(u, x, nu, dx, 1)
    # Compute and return the objective function value
    return 0.5*sum((sol[:,-1]-sol_ref[:,-1])**2)

# Define the constraint function (e.g., x[0] + x[1] <= 1)
def constraints(u):
    # Compute and return a list of constraint values
    return [ui[i]-u[i] for i in range(dimension)]  # Example constraint 2


# Optimize using CMA-ES
es = cma.CMAEvolutionStrategy(dimension * [0], 0.5)
while not es.stop():
    solutions = es.ask()
    objective_values = [objective(x) for x in solutions]
    constraint_values = [constraints(x) for x in solutions]

    # Update Lagrange multipliers and penalty coefficients
    aug_lag.set_coefficients(objective_values, constraint_values)

    # Compute penalized objective values
    penalized_values = [f + sum(aug_lag (g)) for f, g in zip(objective_values, constraint_values)]

    # Update the optimizer
    es.tell(solutions, penalized_values)
    es.disp()