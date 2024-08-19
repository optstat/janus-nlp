import torch
import janus_nlp
import numpy as np
import smac
import cyipopt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario, RunHistory
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter 
from dask.distributed import Client, Queue, get_worker
import matplotlib.pyplot as plt


# Set the device
device = torch.device("cpu")
dtype = torch.double #Ensure that we use double precision

# Define parameter bounds
p10min, p10max = -1000.0, 1000.0 #Currently this is a dummy variable
p20min, p20max = -1000.0, 1000.0
ftmin, ftmax   = 0.01, 0.1

# Define normalization and standardization functions
def normalize(X, bounds):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

def denormalize(X, bounds):
    return X * (bounds[1] - bounds[0]) + bounds[0]

def propagate_state(mu, dt=1.0):
    #          std::tuple<torch::Tensor, torch::Tensor> propagate_state(const torch::Tensor &mu,
    #                                                         const torch::Tensor &x1,
    #                                                         const torch::Tensor &x2,
    #                                                         const torch::Tensor &ft,
    #                                                         const torch::Tensor &params)
    x10t = torch.tensor([[2.0]], dtype=dtype, device=device)
    x20t = torch.tensor([[0.0]], dtype=dtype, device=device)
    mut = torch.tensor([[mu]], dtype=dtype, device=device).squeeze()
    paramst = torch.tensor([1.0e-12, 1.0e-16], dtype=dtype, device=device)
    #Propagate for a long time to get the limit cycle
    resi = janus_nlp.propagate_state(mut, 
                                     x10t, 
                                     x20t, 
                                     torch.tensor([[10.0]], dtype=dtype, device=device), 
                                     paramst)
    x1it = resi[0]
    x2it = resi[1]
    dtt = torch.tensor([[dt]], dtype=dtype, device=device)
    #get the path to the limit cycle
    resf = janus_nlp.propagate_state(mut, x1it, x2it, dtt, paramst)
    x1ft = resf[0]
    x2ft = resf[1]
    return [[x1it, x2it], [x1ft, x2ft]]

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

        jac_dual = janus_nlp.mint_jac_eval(x).squeeze().flatten().numpy()
        #jac_fd = janus_nlp.mint_jac_eval_fd(x).squeeze().flatten().numpy()
        #print(f"Jacobian dual: {jac_dual}")
        #print(f"Jacobian FD: {jac_fd}")
        return jac_dual

class VDPTargetFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        p1 = UniformFloatHyperparameter("p1", lower=p10min, upper=p10max, default_value=0.0)
        p2 = UniformFloatHyperparameter("p2", lower=p20min, upper=p20max, default_value=0.0)
        ft = UniformFloatHyperparameter("ft", lower=ftmin, upper=ftmax, default_value=0.1)
        cs.add_hyperparameters([p1, p2, ft])
        return cs
    
    def init(self, x10, x1f, x20, x2f):
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f

    def train(self, config: Configuration, seed: int = 0):
        """Target function to minimize."""

        x = torch.tensor([config['p1'], config['p2'], config['ft']], dtype=dtype, device=device).unsqueeze(0)
        assert p10min <= config['p1'] <= p10max, "p1 out of bounds"
        assert p20min <= config['p2'] <= p20max, "p2 out of bounds"
        assert ftmin <= config['ft'] <= ftmax, "ft out of bounds"

        #We will use ipopt to solve the optimization problem
        problem = VDPMintIpopt(self.x10, self.x1f, self.x20, self.x2f)

        nlp = cyipopt.Problem(
            n=3,
            m=3,
            problem_obj=problem,
            lb=[p10min, p20min, ftmin],
            ub=[p10max, p20max, ftmax],
            cl=[-0.001, -0.001, -1.0e-6],
            cu=[ 0.001,  0.001,  1.0e-6]
        )

        # Set the options
        nlp.add_option('hessian_approximation', 'limited-memory')  # Enable limited memory BFGS (L-BFGS)
        nlp.add_option('linear_solver', 'mumps')  # Set MUMPS as the linear solver
        nlp.add_option('tol', 1e-4)               # Set the tolerance to 10^-4
        nlp.add_option('print_level', 5)          # Set print level to 5
        nlp.add_option('max_iter', 100)       # Set the maximum number of iterations to 1000
        nlp.add_option('mu_strategy', 'adaptive')  # Set the barrier parameter strategy to adaptive
        nlp.add_option("derivative_test", "first-order")  # Check the gradient
        x0 = np.array([config['p1'], config['p2'], config['ft']])
        x, info = nlp.solve(x0)
        #Check to see if the optimization was successful
        res = 1.0e6
        if info["status"] == 0 or info["status"] == 1:
            res = problem.objective(x)
        print(f"Optimal solution: {res}")            
        return res

def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    plt.figure()

    # Plot all trials
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        p2 = config["p2"]
        ft = config["ft"]
        y = v.cost  # type: ignore
        plt.scatter(p2, y, c="blue", alpha=0.1, zorder=9999, marker="o")
        plt.scatter(ft, y, c="blue", alpha=0.1, zorder=9999, marker="o")
        

    # Plot incumbent
    plt.scatter(incumbent["p2"], incumbent["ft"], c="red", zorder=10000, marker="x")
    print(f"Optimal p2: {incumbent['p2']}, Optimal ft: {incumbent['ft']}")
    plt.xlabel("p2")
    plt.ylabel("Objective Value")
    plt.title("VDP Function Optimization")
    plt.show()

if __name__ == "__main__":
    start_pt, end_pt = propagate_state(0.1, 0.05)
    print(f"Start point: {start_pt}")
    print(f"End point: {end_pt}")
    x10 = start_pt[0]
    x20 = start_pt[1]
    x1f = end_pt[0]
    x2f = end_pt[1]

    model = VDPTargetFunction()
    model.init(x10, x1f, x20, x2f) #Set the initial and final points
    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=500)  # Optionally, increase the number of initial configurations)
    
    initial_design = smac.initial_design.sobol_design.SobolInitialDesign(scenario, n_configs=100)  # 20 initial points


    # Now we use SMAC to find the best hyperparameters
    smac = HPOFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        initial_design=initial_design,
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    #Now pass this to the Newton method to calculate the precise value
    # Ensure that incumbent["p2"] and incumbent["ft"] are sequences
    p1 = incumbent["p1"] if isinstance(incumbent["p1"], (list, tuple)) else [incumbent["p1"]]
    p2 = incumbent["p2"] if isinstance(incumbent["p2"], (list, tuple)) else [incumbent["p2"]]
    ft = incumbent["ft"] if isinstance(incumbent["ft"], (list, tuple)) else [incumbent["ft"]]

    # Convert to torch tensors
    p1_tensor = torch.tensor([p1], dtype=dtype, device=device)
    p2_tensor = torch.tensor([p2], dtype=dtype, device=device)
    ft_tensor = torch.tensor([ft], dtype=dtype, device=device)
    print(f"p1_tensor sizes: {p1_tensor.size()}")
    print(f"p2_tensor sizes: {p2_tensor.size()}")
    print(f"ft_tensor sizes: {ft_tensor.size()}")

    # Pass the tensors to janus_nlp.vdpNewt
    print(f"Starting Newton method")
    res = janus_nlp.vdpNewt(p1_tensor, p2_tensor, ft_tensor, 1.0e-12, 1.0e-14)
    print(f"Optimal solution: {res}")

    # Let's plot it too
    #plot(smac.runhistory, incumbent)
    #Now use this estimate to get a more accurate estimate of the incumbent
    #using a Newton Method

