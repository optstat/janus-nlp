import torch
import numpy as np
import matplotlib.pyplot as plt
import ray
from scipy.integrate import solve_ivp
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.metrics import mean_squared_error

uf 

device = torch.device("cpu")

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

def forward(u : np.ndarray, x : np.ndarray, nu : float, dx : float, ft: float) -> torch.Tensor:
  t_span = [0, ft]
  sol = solve_ivp(dyns_burgers, t_span, u, method='Radau',jac=jac_burgers, rtol=1e-3, atol=1e-6, args=(nu, dx))
  return sol.y

def augmented_objective_function(u0 : np.ndarray, nu : float, ft, lambdap1, mup):
    t_span = (0, ft)
    y0 = [u0]
    sol = solve_ivp(dyns_burgers, t_span, y0, method='Radau', jac = jac_burgers ,rtol=1e-9, atol=1e-12)
    p1fp = sol.y[0,-1]
    p2fp = sol.y[1,-1]
    x1fp = sol.y[2,-1]
    x2fp = sol.y[3,-1]
    print(f"x1fp+|x1fp|: {x1fp+np.abs(x1fp)}")
    print(f"x2fp-x2f: {x2fp-x2f}")
    obj = ft - lambdap1*(x1fp+np.abs(x1fp)) - lambdap2*(x2fp-x2f) +np.log( 1+0.5*mup*(x1fp+np.abs(x1fp))**2 + 0.5*mup*(x2fp-x2f)**2)
    #obj = ft  - lambdap2*(x2fp-x2f) +0.5*mup2*(x2fp-x2f)**2
    return obj







# Function to run SMAC with the initial condition list and the objective function with a seed

def initialize_smac(scenario, overwrite=False):
    # SMAC facade using your objective function
    def optimize_hyperparameters_smac(config, seed=0):
      global mup, u0, nu
      # Convert SMAC configuration to a dictionary
      config = config.get_dictionary()
    
      # Set up LightGBM parameters
      lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        **config
      }
      p1 = config["p1"]
      p2 = config["p2"]
      ft = config["ft"]
      lambdap1 = config["lambdap1"]
      lambdap2 = config["lambdap2"]
      x10 = ic[0]
      x20 = ic[1]
      obj= augmented_objective_function(p1, p2, ft, lambdap1, lambdap2, mup, x10, x20)
      print(f"Optimizing with p1={p1}, p2={p2} and ft={ft} and mup={mup} and initial conditions {ic}", flush=True)    
      print(f"Objective function value: {obj}")
      return obj

    smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=optimize_hyperparameters_smac, overwrite=overwrite)

    # Optimize hyperparameters
    return smac



@ray.remote
def do_optimize(initial_condition):
  global mup, ic
  lambdap = np.asarray([0.0, 0.0]).astype(np.float64)
   
  mup = 0.01
  ic = initial_condition


  count = 0
  p1 = 0.0
  p2 = 0.0
  ft = (ftmin+ftmax)/2
  ########################################################
  #Penalty method with Global Bayesian Optimization 
  ########################################################
  converged = False
  p1_global = p1
  p2_global = p2
  ft_global = ft
  cnorms_global = 100.0
    

  while count < 5:
    count = count + 1
    # Define the configuration space
    # This is a warm start because the defaults are the results of the previous optimization
    cs = ConfigurationSpace(name="vpd config space", space={"p1": Float("p1", bounds=(p10min, p10max), default=p1),
                                                                "p2": Float("p2", bounds=(p20min, p20max), default=p2),
                                                                "ft": Float("ft", bounds=(ftmin, ftmax), default=ft),
                                                                "lambdap1": Constant("lambdap1", value=lambdap[0]),
                                                                "lambdap2": Constant("lambdap2", value=lambdap[1])})
    #The following are lightgbm hyperparameters
    cs.add_hyperparameters([
      UniformFloatHyperparameter("learning_rate", 0.01, 0.3, default_value=0.1),
      UniformIntegerHyperparameter("num_leaves", 10, 100, default_value=31),
      UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100),
      UniformIntegerHyperparameter("min_child_samples", 5, 50, default_value=20),
      UniformFloatHyperparameter("subsample", 0.5, 1.0, default_value=0.8),
      CategoricalHyperparameter("boosting_type", ["gbdt", "dart", "goss"], default_value="gbdt"),
    ])
    # Run SMAC to optimize the objective function
    scenario = Scenario(cs, n_trials=1000)
    optimizer = initialize_smac(scenario, overwrite=True)

    incumbent = optimizer.optimize()
    print(f"Optimal hyperparameters: {incumbent}")
    p1 = incumbent["p1"]
    p2 = incumbent["p2"]
    ft = incumbent["ft"]
    t_span = (0, ft)
    y0 = [p1, p2, ic[0], ic[1]]
    print(f'y0 passed into solve_ivp: {y0}')
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau', jac=dyns_jacobian, rtol=1e-9, atol=1e-9)
    xicst = torch.tensor([ic], dtype=torch.float64, device=device)
    print(f'xicst: {xicst}')
    x0t = torch.tensor([[p1, p2, ft]], dtype=torch.float64, device=device)
    print(f'x0t: {x0t}')
    lambdapt = torch.zeros((1,2), dtype=torch.float64, device=device)
    print(f'lambdapt: {lambdapt}')
    mupt = torch.ones((1,1), dtype=torch.float64, device=device)*mup
    print(f'mupt: {mupt}')
    params = torch.tensor([1.0e-9, 1.0e-9], dtype=torch.float64, device=device)
    print(f'params: {params}')
    #Perform a check of the integrator using Ipopt
    janus_nlp.set_auglangr_x0(xicst[:,0], xicst[:,1])
    janus_nlp.set_ulimits(u1min, u2min, u3min, u1max, u2max, u3max)
    
    [rest, gradst, yend, cst, cnormt, jact] = \
                              janus_nlp.mint_auglangr_propagate(xicst, x0t, lambdapt, mupt, params)

    x1fp = sol.y[2:3,-1].reshape((1,1))
    x2fp = sol.y[3:4,-1].reshape((1,1))
    cs = np.concatenate((x1fp+np.abs(x1fp), x2fp-x2f), axis=1)
    cnorms = np.linalg.norm(cs, ord=2, axis=1, keepdims=True)
    print(f'yend from ipopt: {yend} versus sol.y[:,-1]={sol.y[:,-1]}')
    print(f'Check cnorm from ipopt: {cnormt} versus cnorm from scipy: {cnorms}')

    if cnorms < 1e-3:
      print(f"cnorms converged to {cnorms} count {count} for initial conditions: {initial_condition}")
      converged = True
      break
    else:
      converged = False
      print(f"Applying case 2 cnorms: {cnorms} count {count} initial conditions: {initial_condition}")
      mup=mup*100.0
    #Keep track of the best solution in case the optimization does not converge
    if cnorms < cnorms_global:
      cnorms_global = cnorms
      p1_global = p1
      p2_global = p2
      ft_global = ft
  ########################################################################################################
  #Now implement the full ALM algorithm
  #Propagate the solution to initialize the parameters
  #xopt = phat
  #xic = ics
  if converged:
    lambdap = torch.zeros((1,2), dtype=torch.float64, device=device)
    #We keep mup the same as from the penalty method.
    mup = torch.ones((1,1), dtype=torch.float64, device=device)*mup
    omegap = mup.reciprocal()
    etap = mup.pow(0.1).reciprocal()

    #####################################################
    #Augmented Lagrangian optimization 
    #Convert to tensors
    count = 0
    cnorms = torch.ones((1,1), dtype=torch.float64, device=device)*10.0
    xopt = torch.tensor([[p1, p2, ft]], dtype=torch.float64, device=device)
    print(f"Initial guess: {xopt}")
    print(f"Initial lambdap: {lambdap}")
    print(f"Initial mup: {mup}")
    print(f"Initial omegap: {omegap}")
    print(f"Initial etap: {etap}")
    
    while (cnorms > tol_cnorm).any() and count < 5:
      #Need to convert to batch tensors
      xics = torch.tensor([ic], dtype=torch.float64, device=device)
      omegap, xopt, grads, cs, cnorms, jac= batched_augLang_ipopt(xics, xopt, lambdap, mup, omegap.mean(), target_value=tol_omega)
      if (omegap < tol_omega and cnorms < tol_cnorm ).all():
        print(f'Finished optimization')
        break
      #Keep track of the best solution
      if cnorms.item() < cnorms_global:
        cnorms_global = cnorms.item()
        p1_global = xopt[0,0].item()
        p2_global = xopt[0,1].item()
        ft_global = xopt[0,2].item()
      else:
        break #We are not improving so we can stop
         

      count = count + 1
      print(f"Objective function value from ipopt: {omegap}")
      print(f"cnorms from ipopt: {cnorms}")
      print(f"etap: {etap}")
      print(f"xopt from ipopt: {xopt}")
      print(f"omegap from ipopt: {omegap}")
      m = (cnorms < etap).any(dim=1)

      if m.any():
        print(f'Applying case 1: Modifying the langrange multipliers')
        mupls = torch.einsum("mi,m->m",lambdap, mup.squeeze(1) )
        lambdap[m]=lambdap[m]-mup[m]*cs[m]/(1.0+mupls)
        etap[m]=etap[m]/mup[m].pow(0.9)
        omegap[m]=omegap[m]/mup[m]
        print(f"lambdap: {lambdap}")
        print(f"etap: {etap}")
        print(f"omegap: {omegap}")
        print(f"mup: {mup}")
      else:
        print(f'Applying case 2: Increasing the penalty')
        mup[~m]=mup[~m]*100.0
        etap[~m]=mup[~m].pow(0.1).reciprocal()
        omegap[~m]=mup[~m].reciprocal()
        print(f"mup: {mup}")
        print(f"lambdap: {lambdap}")
        print(f"etap: {etap}")
        print(f"omegap: {omegap}")
  print(f'Optimization finished with cnorms_global={cnorms_global}')  
  return p1_global, p2_global, ft_global, converged

def augmented_opt(N:int, iteration=0, numSamples=2):
    # Define bounds for the 2D Sobol sequence
    lower_bounds = torch.tensor([-1.0 for _ in range(N)], device=device, dtype=torch.float64)
    upper_bounds = torch.tensor([1.0 for _ in range(N)], device=device, dtype=torch.float64)
    #Need to append nu
    lower_bounds = torch.cat((lower_bounds, torch.tensor([0.001], dtype=torch.float64, device=device)))
    upper_bounds = torch.cat((upper_bounds, torch.tensor([0.1], dtype=torch.float64, device=device)))
    sobol = torch.quasirandom.SobolEngine(dimension=N+1)
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

def main():
    global uf
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
    uf = sol[:,-1]  #This is now the data to be used for the optimization

    #plot the solution as a 2D plot
    plt.figure()
    plt.plot(x,sol)
    plt.savefig('./images/burgers.png')
    plt.close()
    plt.figure()
    plt.plot(sol[:,-1])
    plt.savefig('./images/burgers_final.png')
    plt.close()
   


if __name__ == "__main__":
  main()