import torch
import janus_nlp
import numpy as np
import cyipopt
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll  # Correct function name
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

# Set the device
device = torch.device("cpu")
dtype = torch.double  # Ensure that we use double precision

# Define parameter bounds
p10min, p10max = -0.1, 0.1
p20min, p20max = -0.1, 0.1
mu = 1.0
ftmin, ftmax = 0.1, 10.0

# Define normalization and standardization functions
def normalize(X, bounds):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

def denormalize(X, bounds):
    return X * (bounds[1] - bounds[0]) + bounds[0]


def vdp(x, x10, x1f, x20, x2f):
    p1 = x[:, 0:1]
    print (f"p1: {p1}")
    p2 = x[:, 1:2]
    print(f"p2: {p2}")    

    ft = x[:, 2:3]    
    print(f"ft: {ft}")
    x10t = torch.ones_like(p2) * x10
    x20t = torch.ones_like(p2) * x20
    x1ft = torch.ones_like(p2) * x1f
    x2ft = torch.ones_like(p2) * x2f
    janus_nlp.set_x0(x10t, x20t)
    janus_nlp.set_xf(x1ft, x2ft)
    janus_nlp.mint_set_mu(mu)
    errors = janus_nlp.vdp_solve(x)    
    return errors


class VDPMintIpopt(cyipopt.Problem):
    def __init__(self, x10, x1f, x20, x2f):
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f
        self.x10t = torch.tensor([[self.x10]], dtype=dtype, device=device)
        self.x20t = torch.tensor([[self.x20]], dtype=dtype, device=device)
        self.x1ft = torch.tensor([[self.x1f]], dtype=dtype, device=device)
        self.x2ft = torch.tensor([[self.x2f]], dtype=dtype, device=device)



    def objective(self, x):
        ft = x[2]
        return ft
    
    def gradient(self, x):
        grad = np.zeros(3)
        grad[2] = 1.0
        return grad
    
    def constraints(self, x):
        p1 = x[0]
        p2 = x[1]
        ft = x[2]
        x = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
        janus_nlp.set_mint_x0(self.x10t, self.x20t)
        janus_nlp.set_mint_xf(self.x1ft, self.x2ft)
        janus_nlp.mint_set_mu(mu)
        errors = janus_nlp.mint_vdp_solve(x).squeeze().flatten().numpy()
        print(f"Errors: {errors}")
        return errors
    
    def jacobian(self, x):
        p1 = x[0]
        p2 = x[1]
        ft = x[2]
        x = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
        janus_nlp.set_mint_x0(self.x10t, self.x20t)
        janus_nlp.set_mint_xf(self.x1ft, self.x2ft)
        janus_nlp.mint_set_mu(mu)
        #jac_dual = janus_nlp.mint_jac_eval(x).squeeze().flatten().numpy()
        jac_fd = janus_nlp.mint_jac_eval_fd(x).squeeze().flatten().numpy()
        #print(f"Jacobian dual: {jac_dual}")
        #print(f"Jacobian FD: {jac_fd}")
        return jac_fd

class VDPTargetFunction:
    def __init__(self, x10, x1f, x20, x2f):
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f

    def train(self, x):
        """Target function to minimize."""

        #We will use ipopt to solve the optimization problem
        problem = VDPMintIpopt(self.x10, self.x1f, self.x20, self.x2f)

        nlp = cyipopt.Problem(
            n=3,
            m=3,
            problem_obj=problem,
            lb=[p10min, p20min, ftmin],
            ub=[p10max, p20max, ftmax],
            cl=[-0.001, -0.001, -1.0e-3],
            cu=[ 0.001,  0.001,  1.0e-3]
        )

        # Set the options
        nlp.add_option('hessian_approximation', 'limited-memory')  # Enable limited memory BFGS (L-BFGS)
        nlp.add_option('linear_solver', 'mumps')  # Set MUMPS as the linear solver
        nlp.add_option('tol', 1e-4)               # Set the tolerance to 10^-4
        nlp.add_option('print_level', 5)          # Set print level to 5
        nlp.add_option('max_iter', 1000)       # Set the maximum number of iterations to 1000
        nlp.add_option('mu_strategy', 'adaptive')  # Set the barrier parameter strategy to adaptive
        nlp.add_option("derivative_test", "first-order")  # Check the gradient
        x0 = x.squeeze().flatten().numpy()
        x, info = nlp.solve(x0)
        #Check to see if the optimization was successful
        res = 1.0e6
        if info["status"] == 0 or info["status"] == 1:
            res = problem.objective(x)
        print(f"Optimal solution: {res}")            
        return res



# Initialize the VDP Target Function
model = VDPTargetFunction(x10=1.0, x1f=-1.0, x20=5.0, x2f=-2.0)

# Example bounds for three parameters
p10min, p10max = -0.1, 0.1
p20min, p20max = -0.1, 0.1
ftmin, ftmax   = 0.1, 10.0

# Define the bounds as a tensor
bounds = torch.tensor([
    [p10min, p20min, ftmin],  # Lower bounds
    [p10max, p20max, ftmax]   # Upper bounds
], dtype=torch.float64, device=device)

#Generate a Sobol sequence between the bounds
n_samples = 10
# Create a Sobol sequence generator
sobol = qmc.Sobol(d=3, scramble=False)

# Generate 1 samples
n = 1
samples = sobol.random(n)
# Rescale the samples to the desired bounds
lower_bounds = bounds[0]
upper_bounds = bounds[1]

# The scaling formula is:
# scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * samples
scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * samples


X_train = torch.tensor(scaled_samples, dtype=dtype, device=device)

print(f'X_train shape: {X_train.shape}')
Y_train = model.train(X_train.squeeze())
print(f'Y_train shape: {Y_train.shape}')
exit(1)

# Define the Gaussian Process model
gp_model = SingleTaskGP(X_train, Y_train)
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(mll)  # Correct function to fit the model

# Define the acquisition function
acqf = LogExpectedImprovement(model=gp_model, best_f=Y_train.min())

# Optimization loop
n_iterations = 50
for i in range(n_iterations):
    # Optimize the acquisition function to find the next candidate
    candidates, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    # Evaluate the candidate
    new_X = candidates.detach()
    new_Y = vdp_function.evaluate(new_X)

    # Update the training data
    X_train = torch.cat([X_train, new_X])
    Y_train = torch.cat([Y_train, new_Y])

    # Update the model with new data
    gp_model.set_train_data(X_train, Y_train, strict=False)
    fit_gpytorch_mll(mll)

    # Update the acquisition function with the new best value
    acqf = LogExpectedImprovement(model=gp_model, best_f=Y_train.min())

# Get the best candidate found
best_candidate = X_train[Y_train.argmin()]
best_value = Y_train.min()

print(f"Best candidate: {best_candidate}")
print(f"Best value: {best_value}")
