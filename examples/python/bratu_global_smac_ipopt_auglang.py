import torch
import numpy as np
import cyipopt
import matplotlib.pyplot as plt
import ray
import time
import pickle

import janus_nlp
from scipy.integrate import solve_ivp
from ConfigSpace import ConfigurationSpace, Float, Constant
from smac import HyperparameterOptimizationFacade, Scenario

# Set the device
device = torch.device("cpu")
dtype = torch.double  # Ensure that we use double precision
#Global variables
#Start point in the fast region
x10 = 0.0
#End point in the slow region
x1f = 0.0
lam = 3.51
ft = 1.0




class BratuAugPMPIpopt(cyipopt.Problem):
    def __init__(self, xics, lambdap, mup, x0):
        self.M = xics.size(0) #Number of samples
        self.xics = xics
        self.rtol = 1.0e-9
        self.atol = 1.0e-9
        self.lambdap = lambdap
        self.mup = mup
        self.obj = 1.0e10
        self.sol = x0
    

    def objective(self, x):
      xt = torch.tensor(x).reshape(self.M,1)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

      janus_nlp.linear_bratu_set_x0(self.xics[:,0])
      janus_nlp.linear_bratu_set_xf(x1f)
      janus_nlp.linear_minu_set_lam(lam)
 
      [obj, grads, errors, errors_norm, jac] = janus_nlp.bratu_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)
      res = obj.sum().flatten().numpy()
      if res < self.obj:
         self.obj = res
         self.sol = x
      print(f"Objective: {obj}")

      return res
    
    def gradient(self, x):
      xt = torch.tensor(x).reshape(self.M,1)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

      janus_nlp.linear_bratu_set_x0(self.xics[:,0])
      janus_nlp.linear_bratu_set_xf(x1f)
      janus_nlp.linear_bratu_set_lam(lam)
 
      [obj, grads, errors, errors_norm, jac] = janus_nlp.bratu_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)



      print(f"Gradients: {grads}")
      return grads.squeeze().flatten().numpy()
    






#The inputs are all tensors
def batched_augLang_ipopt(xics, x, lambdap, mup, tol):
  # Example bounds for three parameters
  # Define parameter bounds-there are not necessarily the same 
  # as the ones used in the outer optimization
  # The job of the outer optimization is to find the best initial guess
  # for the inner optimization so the inner optimization can have wider parameters

  print(f"Setting up the problem with initial conditions {xics}.")
  M = x.size(0)
  print(f"Setting up the problem with sample size {M}.")
  
  res     = torch.zeros((M,1), dtype=torch.float64, device=device)
  sols    = torch.zeros((M,3), dtype=torch.float64, device=device)
  grads   = torch.zeros((M,3), dtype=torch.float64, device=device)
  cs      = torch.zeros((M,5), dtype=torch.float64, device=device)
  cnorm   = torch.zeros((M,), dtype=torch.float64, device=device)
  problem = BratuAugPMPIpopt(xics, lambdap, mup, x.flatten().numpy())

  print(f'problem = {problem}')
  xlb = [0.0 for _ in range(M)]
  xub = [0.0 for _ in range(M)]
  #Limit the exploration space
  for i in range(M):
    if x[i,0] < 0: 
      xlb[i]   = x[i,0]*2.0
      xub[i]   = x[i,0]*0.5
    else:
      xlb[i]   = x[i,0]*0.5
      xub[i]   = x[i,0]*2.0



  x0 = x.flatten().numpy().tolist()
  print(f"Initial guess: {x0}")
  print(f'len(x0): {len(x0)}')
  print(f'len(xlb): {len(xlb)}')
  print(f'len(xub): {len(xub)}')
  params = torch.tensor([problem.rtol, problem.atol], dtype=torch.float64)
  
  [res, grads, cs, cnorm, jac] = \
                              janus_nlp.bratu_auglangr_propagate(problem.xics, x, problem.lambdap, problem.mup, params)

  scaling_factor = 1.0/res.abs().item()
  print(f'Problem scaling_factor={scaling_factor}')
  nlp = cyipopt.Problem(
            n=M,
            m=0,
            problem_obj=problem,
            lb=xlb,
            ub=xub,
            cl=[],
            cu=[],
        )
  if isinstance(tol, torch.Tensor):
      acc_tol=float(tol.item())
  else:
      acc_tol=float(tol)
  #This value cannot be too small or ipopt will not converge
  if acc_tol < 1.0e-6:
      acc_tol = 1.0e-6
  # Set the options
  nlp.add_option('hessian_approximation', 'limited-memory')  # Enable limited memory BFGS (L-BFGS)
  nlp.add_option('nlp_scaling_method', 'gradient-based') #Use a gradient based method for scaling
  nlp.add_option('obj_scaling_factor', scaling_factor) #Set the scaling factor
  nlp.add_option('linear_solver', 'mumps')  # Set MUMPS as the linear solver
  nlp.add_option('acceptable_tol', acc_tol)  # Set tolerance for acceptable objective value
  nlp.add_option('acceptable_iter', 0)  # Allow IPOPT to stop immediately when an acceptable solution is found
  nlp.add_option('print_level', 5)          # Set print level to 5
  nlp.add_option('max_iter', 5)       # Set the maximum number of iterations.  This should converge very quickly 
  nlp.add_option('mu_strategy', 'adaptive')  # Set the barrier parameter strategy to adaptive
  #nlp.add_option("derivative_test", "first-order")  # Check the gradient
  
  sol, info = nlp.solve(x0)
  if info['status'] != 0 and info['status'] != 1:
     #Get the best solution arrived at during the iterations
     sol = problem.sol
  print(f"Received info: {info}")
  sol = torch.tensor(sol).reshape((M,3))
 
  [res, grads, cs, cnorm, jac] = \
                              janus_nlp.mint_auglangr_propagate(problem.xics, sol, problem.lambdap, problem.mup, params)

  return res, sol, grads, cs, cnorm, jac






  


def dyns_aug(t, y, lam):
    x1 = y[0]
    x2 = y[1]
    J  = y[2]
    dx1dt = x2
    dx2dt = -lam*np.exp(x1)
    dJdt  = 0.5*x1*x1
    return np.stack([dx1dt, dx2dt, dJdt], axis=-1)



def dyns_jacobian(t, y, lam):
    x1 = y[0]
    x2 = y[1]
    J  = y[2]

    jac = np.zeros((3, 3))
    jac[0,1]= 1.0
    jac[1,0]= -lam*np.exp(x1)
    jac[2,0]= x1xx
    return jac



def augmented_objective_function(ft, mup, x10):
    t_span = (0, ft)
    y0 = [x10, 0.0]
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau', jac = dyns_jacobian ,rtol=1e-9, atol=1e-12, args=(lam,))
    x1fp = sol.y[0,-1]
    x2fp = sol.y[1,-1]
    print(f"x1fp-x1f: {x1fp-x1f}")
    obj = Jfp + np.log( 1+0.5*mup*(x1fp-x1f)**2)
    return obj





# Function to run SMAC with the initial condition list and the objective function with a seed
def run_smac_with_initial_conditions(scenario, initial_condition):

    # Adjust your objective function if necessary to incorporate initial conditions and seeds
    def optimize_hyperparameters_smac(config, seed=0):
      p1 = config["p1"]
      mup = config["mup"]
      x10 = initial_condition[0]
      obj= augmented_objective_function(p1, ft, mup, x10)
      print(f"Optimizing with p1={p1}, and ft={ft}")
      print(f"Objective function value: {obj}")
      return obj


    # SMAC facade using your objective function
    smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=optimize_hyperparameters_smac, overwrite=True)

    # Optimize hyperparameters
    return smac.optimize()

@ray.remote
def do_optimize(initial_conditions):
  mup = 0.01

  count = 0
  p1 = 0.0
  ########################################################
  #Penalty method with Global Bayesian Optimization 
  ########################################################
  converged = False
  while count < 10:
    count = count + 1
    
    # Define the configuration space
    cs = ConfigurationSpace(name="vpd config space", space={"p1": Float("p1", bounds=(p10min, p10max), default=p1),
                                                            "mup": Constant("mup", value=mup) })
    
    # Run SMAC to optimize the objective function
    scenario = Scenario(cs, deterministic=False, n_trials=5000)
    incumbent = run_smac_with_initial_conditions(scenario, initial_conditions)
    print(f"Optimal hyperparameters: {incumbent}")
    p1 = incumbent["p1"]
    t_span = (0, ft)
    y0 = [p1, initial_conditions[0], 0.0]
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau',rtol=1e-6, atol=1e-9)
    x1fp = sol.y[0,-1]
    cs = np.asarray([x1fp-x1f])
    cnorms = np.linalg.norm([x1fp-x1f])
    if cnorms < 1e-3:
      print(f"cnorms converged to {cnorms} count {count} for initial conditions: {initial_conditions}")
      converged = True
      break
    else:
      converged = False
      print(f"Applying case 2 cnorms: {cnorms} count {count} initial conditions: {initial_conditions}")
      mup=mup*100.0
  return p1, converged

def augmented_opt(iteration=0, numSamples=2):
    # Define bounds for the 2D Sobol sequence
    lower_bounds = torch.tensor([-1.0], device=device, dtype=torch.float64)
    upper_bounds = torch.tensor([1.0], device=device, dtype=torch.float64)
    sobol = torch.quasirandom.SobolEngine(dimension=1)
    # Number of initial conditions to generate
    num_initial_conditions = numSamples #Same as the number of samples
    sobol.fast_forward(iteration * numSamples)  
    # Generate Sobol sequence and scale it to desired range
    sobol_sequence = sobol.draw(num_initial_conditions).double()
    initial_conditions = lower_bounds + (upper_bounds - lower_bounds) * sobol_sequence
    # Convert Sobol sequence to a list of lists for processing
    initial_conditions_list = initial_conditions.tolist()
    print(f'Initial conditions: {initial_conditions_list}')
    
    result_ids = [do_optimize.remote(cond) for cond in initial_conditions_list]
    results = ray.get(result_ids)
    resultst = torch.tensor(results, dtype=torch.float64)
    print(f"Results: {results}")
    initial_conditionst = torch.tensor(initial_conditions_list, dtype=torch.float64) 
    #return penalty_function(p1, p2, ft, penalty=penalty)
    return initial_conditionst, resultst
    
if __name__ == "__main__":
  ray.init()
  #Modify the iteration number to generate different initial conditions
  ics, phat = augmented_opt(0, 100)
  ray.shutdown()
  print(f"Initial conditions: {ics}")
  print(f"BO results: {phat}")
  #Check the answer by forward propagation
  M = ics.size(0)
  for i in range(M):
    x10 = ics[i,0].item()
    p10 = phat[i,0].item()
    t_span = (0, ft)
    y0 = [p10, x10, 0.0]
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau', jac = dyns_jacobian ,rtol=1e-9, atol=1e-12)
    p1fp = sol.y[0,-1]
    x1fp = sol.y[1,-1]
    Jfp = sol.y[2,-1]
    print(f"x1fp-x1f for sample {i}: {x1fp-x1f}")
  
  
