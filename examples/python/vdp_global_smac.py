import torch
import janus_nlp
import numpy as np
import smac
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario, RunHistory
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter 
from dask.distributed import Client, Queue, get_worker
import matplotlib.pyplot as plt


# Set the device
device = torch.device("cpu")
dtype = torch.double #Ensure that we use double precision

# Define parameter bounds
p10min, p10max = -1.0, 1.0
p20min, p20max = -1.0, 1.0
ftmin, ftmax   = 0.1, 8.0
x1f, x2f       = 1.8,  1.5
x10, x20       = 1.8,  1.5

# Define normalization and standardization functions
def normalize(X, bounds):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

def denormalize(X, bounds):
    return X * (bounds[1] - bounds[0]) + bounds[0]

def vdp(x):
    p1 = x[:, 0:1]
    print (f"p1: {p1}")
    p2 = x[:, 1:2]
    print(f"p2: {p2}")    

    ft = x[:, 2:3]    
    print(f"ft: {ft}")
    x10t = torch.ones_like(p2) * x10
    x20t = torch.ones_like(p2) * x20
    janus_nlp.set_x0(x10t, x20t)
    x1ft = torch.ones_like(p2) * x1f
    x2ft = torch.ones_like(p2) * x2f
    janus_nlp.set_xf(x1ft, x2ft)
    errors = janus_nlp.vdp_solve(x)    
    return errors

class VDPTargetFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        p1 = UniformFloatHyperparameter("p1", lower=p10min, upper=p10max, default_value=0.0)
        p2 = UniformFloatHyperparameter("p2", lower=p20min, upper=p20max, default_value=0.0)
        ft = UniformFloatHyperparameter("ft", lower=ftmin, upper=ftmax, default_value=7.0)
        cs.add_hyperparameters([p1, p2, ft])

        return cs

    def train(self, config: Configuration, seed: int = 0):
        """Target function to minimize."""

        x = torch.tensor([config['p1'], config['p2'], config['ft']], dtype=dtype, device=device).unsqueeze(0)
        assert p10min <= config['p1'] <= p10max, "p1 out of bounds"
        assert p20min <= config['p2'] <= p20max, "p2 out of bounds"
        assert ftmin <= config['ft'] <= ftmax, "ft out of bounds"

        y = vdp(x)
        return float(y.item())

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
    model = VDPTargetFunction()
    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=10)  # Optionally, increase the number of initial configurations)
    
    initial_design = smac.initial_design.sobol_design.SobolInitialDesign(scenario, n_configs=10)  # 20 initial points


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
    res = janus_nlp.vdpNewt(p1_tensor, p2_tensor, ft_tensor, 1.0e-12, 1.0e-14)
    print(f"Optimal solution: {res}")

    # Let's plot it too
    #plot(smac.runhistory, incumbent)
    #Now use this estimate to get a more accurate estimate of the incumbent
    #using a Newton Method

