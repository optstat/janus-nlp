import cyipopt
import numpy as np

class AugmMerit(cyipopt.Problem):
    def __init__(self, mup, lambdap=0.0):
        super().__init__(n=1, m=0, lb=[-10.0], ub=[10.0])
        self.mu = mup
        self.lam = lambdap 

    def objective(self, x):
        obj = np.log(1.0+0.5*self.mu*(np.sin(5*x[0])-x[0])**2)+self.lam*(np.sin(5*x[0])-x[0])
        return obj

    def gradient(self, x):
      numerator = self.mu * (np.sin(5*x[0]) - x[0]) * (5*np.cos(5*x[0]) - 1)
      denominator = 1.0 + 0.5 * self.mu * (np.sin( 5*x[0]) - x[0]) ** 2
      return numerator / denominator+self.lam*5*np.cos(5*x[0])-self.lam

    def constraints(self, x):
      return np.array([])  # Empty since there are no constraints

    def jacobian(self, x):
      return np.array([])  # Empty since there are no constraints


    def finalize(self, alg_mod, status):
        print("Problem status: %s" % status)

def main():
    mu = 0.01
    lambdap = 0.0
    x0 = np.random.uniform(-10, 10, 1)
    obj_min = np.inf
    count = 0
    while np.abs((np.sin(5*x0)-x0)) > 1e-6:
        count += 1
        print("Solving problem with mu=%g" % (mu))
        problem = AugmMerit(mu, lambdap)
        problem.add_option('mu_strategy', 'adaptive')
        problem.add_option('tol', 1e-4)
        #problem.add_option("line_search_method", "gradient-based")
        #problem.add_option('tol', 1e-4)
        problem.add_option('print_level', 3)
        problem.add_option('max_iter', 10000)
        #problem.add_option('hessian_approximation', 'limited-memory')
        sol, info= problem.solve(x0)
        cs = (np.sin(5*sol[0])-sol[0])
        if np.abs(cs) < 1e-6:
              print("Solution of the primal variables: x=%s\n" % sol)
              print(f'cs={cs}')
              print(f'lambda={lambdap}')
              break
        else:   
           x0 = sol+np.random.uniform(-10, 10, 1)/count

           mu = mu* 10
           print(f'cs={cs}')
           #lambdap = lambdap - mu*cs
           print(f'lambda={lambdap}')  
    print(f'count={count}')
    print(f'cs={cs}')      
    print(f'lambda={lambdap}')
    print(f'mu={mu}') 
        
    
if __name__ == '__main__':
    main()
