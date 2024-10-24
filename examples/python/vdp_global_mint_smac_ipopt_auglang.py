import torch
import numpy as np
import cyipopt
import matplotlib.pyplot as plt
import ray
import time
import pickle

import janus_nlp
from torch.autograd import gradcheck

#Implementation of the Augmented Lagrangian function using PyTorch
#and Adam as the optimizer

#from botorch.models import SingleTaskGP
#from botorch.fit import fit_gpytorch_mll  # Correct function name

from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound
from scipy.integrate import solve_ivp
from botorch.acquisition.objective import ConstrainedMCObjective
from ConfigSpace import ConfigurationSpace, Float, Constant
from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import LCB
from smac.initial_design import SobolInitialDesign

# Set the device
device = torch.device("cpu")
dtype = torch.double  # Ensure that we use double precision
#Global variables
#Start point in the fast region
x10 = 2.0
x20 = 20.0
#End point in the slow region
x1f = -2.0
x2f =  0.0
mu = 10.0
W = 100.0
u1max = 0.0
u1min = 0.0
u2max = 100.0
u2min = 1.0
u3min = 0.0
u3max = 0.0
p10min, p10max = -5.0, 5.0
p20min, p20max = -5.0, 5.0
ftmin, ftmax   = 1.0, 25.0


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
    janus_nlp.set_auglangr_x0(x10t, x20t)
    janus_nlp.set_auglangr_xf(x1ft, x2ft)
    janus_nlp.mint_set_mu(mu)
    errors = janus_nlp.vdp_solve(x)    
    return errors



class VdpNecCond(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, x10, x20, x1f, x2f, lambdap, mup):  
        # Save input for backward pass
        # Perform your custom forward computation
        p2 = input[:, 1:2]
        x10t = torch.ones_like(p2) * x10
        x20t = torch.ones_like(p2) * x20
        x1ft = torch.ones_like(p2) * x1f
        x2ft = torch.ones_like(p2) * x2f

        janus_nlp.set_auglangr_x0(x10t, x20t)
        janus_nlp.set_auglangr_xf(x1ft, x2ft)
        janus_nlp.set_auglangr_mu(mu)
        #            std::tuple<torch::Tensor, torch::Tensor> mint_auglangr_propagate(const torch::Tensor &x,
        #                                                                     const torch::Tensor &lambdap,
        #                                                                     const torch::Tensor &mup,           
        #                                                                     const torch::Tensor &params)
        params = torch.tensor([1.0e-6, 1.0e-9], dtype=torch.float64)
        output, cm, grads, jac = janus_nlp.mint_auglangr_propagate(input, lambdap, mup, params)
        ctx.save_for_backward(grads.clone())
        #ctx.save_for_backward(input)
        #output = input ** 2  # Simple operation
        return output.clone(), cm.clone()

    @staticmethod
    def backward(ctx, grad_output, grad_cm):
        # Retrieve the saved input
        grads,  = ctx.saved_tensors
        # Define the custom gradient
        #print(f"grad_output: {grad_output}")
        #print(f"grads: {grads}")

        return grads*grad_output, None, None, None, None, None, None

class VDPMintIpopt(cyipopt.Problem):
    def __init__(self, x10, x1f, x20, x2f, mu, W):
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f
        self.x10t = torch.tensor([[self.x10]], dtype=dtype, device=device)
        self.x20t = torch.tensor([[self.x20]], dtype=dtype, device=device)
        self.x1ft = torch.tensor([[self.x1f]], dtype=dtype, device=device)
        self.x2ft = torch.tensor([[self.x2f]], dtype=dtype, device=device)
        self.W = W
        self.mu = mu
        self.rtol = 1.0e-6
        self.atol = 1.0e-6
        janus_nlp.set_auglangr_x0(self.x10t, self.x20t)
        janus_nlp.set_auglangr_xf(self.x1ft, self.x2ft)
        janus_nlp.set_auglangr_mu(mu)
        janus_nlp.set_auglangr_W(W)
    

    def objective(self, x):
      p2 = x[0]
      ft = x[1]
      p2t = torch.tensor([[p2]], dtype=dtype, device=device)
      #H = p1*x2+p2*mu*(1-x1**2)*x2-p2*x1+p2*ustar+0.5*W*ustar*ustar+1
      ustar, p1t = janus_nlp.calc_ustar_p1(p2t, self.x10t, self.x20t)
      p1 = p1t.item()
      xt = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

      [obj, grads, errors, jac] = janus_nlp.mint_propagate(xt, params)

      return obj.flatten().numpy()
    
    def gradient(self, x):
      p2 = x[0]
      ft = x[1]
      p2t = torch.tensor([[p2]], dtype=dtype, device=device)

      ustar, p1t = janus_nlp.calc_ustar_p1(p2t, self.x10t, self.x20t)
      p1 = p1t.item()
      xt = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

      [obj, grads, errors, jac] = janus_nlp.mint_propagate(xt, params)

      return grads[:,1:].flatten().numpy()

    
    def constraints(self, x):
        p2 = x[0]
        ft = x[1]
        p2t = torch.tensor([[p2]], dtype=dtype, device=device)

        ustar, p1t = janus_nlp.calc_ustar_p1(p2t, self.x10t, self.x20t)
        p1 = p1t.item()
        xt = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
        params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

        [obj, grads, errors, jac] = janus_nlp.mint_propagate(xt, params)
        print(f"Errors: {errors}")
        return errors[:,:2].flatten().numpy()
    
    def jacobian_exact(self, x):
        p2 = x[0]
        ft = x[1]
        p2t = torch.tensor([[p2]], dtype=dtype, device=device)

        ustar, p1t = janus_nlp.calc_ustar_p1(p2t, self.x10t, self.x20t)
        p1 = p1t.item()
        xt = torch.tensor([p1, p2, ft], dtype=dtype, device=device).unsqueeze(0)
        params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

        [obj, grads, errors, jac] = janus_nlp.mint_propagate(xt, params)
        print(f'Jacobian: {jac}')
        jacl = jac[:,:,1:].squeeze().flatten().numpy()
        print(f"Returning Jacobian: {jacl}")
        return jacl

    def jacobian(self, x):
        p2 = x[0]
        ft = x[1]
        xc = x.copy()
        jac = np.zeros((2,2))
        h = 1.0e-8
        xph = xc.copy()
        xph[0] = xc[0]+h
        gph = self.constraints(xph)
        xmh = xc.copy()
        xmh[0] = xc[0]-h
        gmh = self.constraints(xmh)
        jac[:,0] = (gph-gmh)/(2.0*h)

        h = 1.0e-8
        xph = xc.copy()
        xph[1] = xc[1]+h
        gph = self.constraints(xph)
        xmh = xc.copy()
        xmh[1] = xc[1]-h
        gmh = self.constraints(xmh)
        jac[:,1] = (gph-gmh)/(2.0*h)
        return jac


class VDPAugPMPIpopt(cyipopt.Problem):
    def __init__(self, xics, lambdap, mup, x0):
        self.M = xics.size(0) #Number of samples
        self.xics = xics
        self.mu = mu
        self.rtol = 1.0e-9
        self.atol = 1.0e-9
        self.lambdap = lambdap
        self.mup = mup
        self.obj = 1.0e10
        self.sol = x0
    

    def objective(self, x):
      xt = torch.tensor(x).reshape(self.M,3)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)

      janus_nlp.set_auglangr_x0(self.xics[:,0], self.xics[:,1])
      janus_nlp.set_auglangr_mu(mu)
      janus_nlp.set_auglangr_W(W)
      janus_nlp.set_ulimits(u1min, u2min, u3min, u1max, u2max, u3max)
 
      [obj, grads, errors, errors_norm, jac] = janus_nlp.mint_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)
      res = obj.sum().flatten().numpy()
      if res < self.obj:
         self.obj = res
         self.sol = x
      print(f"Objective: {obj}")

      return res
    
    def gradient(self, x):
      xt = torch.tensor(x).reshape(self.M,3)
      params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)
      janus_nlp.set_auglangr_x0(self.xics[:,0], self.xics[:,1])
      janus_nlp.set_auglangr_mu(mu)
      janus_nlp.set_auglangr_W(W)
      janus_nlp.set_ulimits(u1min, u2min, u3min, u1max, u2max, u3max)

 
      [obj, grads, errors, errors_norm, jac] = janus_nlp.mint_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)
      print(f"Gradients: {grads}")
      return grads.squeeze().flatten().numpy()
    
    #def constraints(self, x):
    #  pass
      #xt = torch.tensor(x).reshape(self.M,3)
      #params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)
      #janus_nlp.set_auglangr_x0(self.xics[:,0], self.xics[:,1])
      #janus_nlp.set_auglangr_mu(mu)
      #janus_nlp.set_auglangr_W(W)
      #janus_nlp.set_ulimits(u1min, u2min, u3min, u1max, u2max, u3max)

      #[obj, grads, errors, errors_norm, jac] = janus_nlp.mint_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)
      
      #gs= torch.flatten(errors).tolist()
      #print(f"Constraints: {gs}")
      #return gs
    
    
    #def jacobianstructure(self):
    #  pass
      # For example, assume you have a 3x3 Jacobian with non-zero entries in:
      # (0, 0), (0, 2), (1, 1), and (2, 0)
      #rows=[]
      #cols=[]
      #for k in range(self.M):
      #  for i in range(2):
      #    for j in range(3):
      #      rows.append(2*k+i)
      #      cols.append(3*k+j)
      #print(f"Jacobian structure: {rows}, {cols}")
      #return (rows, cols)
    
    #def jacobian(self, x):
    #  pass
      #xt = torch.tensor(x).reshape(self.M,3)
      #params = torch.tensor([self.rtol, self.atol], dtype=torch.float64)
      #janus_nlp.set_auglangr_x0(self.xics[:,0], self.xics[:,1])
      #janus_nlp.set_auglangr_mu(mu)
      #janus_nlp.set_auglangr_W(W)
      #janus_nlp.set_ulimits(u1min, u2min, u3min, u1max, u2max, u3max)

      #[obj, grads, errors, errors_norm, jac] = janus_nlp.mint_auglangr_propagate(self.xics, xt, self.lambdap, self.mup, params)
      #jacres = torch.flatten(jac).tolist()
      #print(f"Jacobian: {jacres}")
      #return jacres #This should be fine because it is row ordered

    def jacobian_fd(self, x):
        p1 = x[0]
        p2 = x[1]
        ft = x[2]
        xc = x.copy()
        jac = np.zeros((3,3))

        h = 1.0e-8*max(1.0, abs(xc[0]))
        xph = xc.copy()
        xph[0] = xc[0]+h
        gph = self.constraints(xph)
        xmh = xc.copy()
        xmh[0] = xc[0]-h
        gmh = self.constraints(xmh)
        jac[:,0] = (gph-gmh)/(2.0*h)

        h = 1.0e-8*max(1.0, abs(xc[1]))
        xph = xc.copy()
        xph[1] = xc[1]+h
        gph = self.constraints(xph)
        xmh = xc.copy()
        xmh[1] = xc[1]-h
        gmh = self.constraints(xmh)
        jac[:,1] = (gph-gmh)/(2.0*h)
        
        h = 1.0e-8*max(1.0, abs(xc[2]))
        xph = xc.copy()
        xph[2] = xc[2]+h
        gph = self.constraints(xph)
        xmh = xc.copy()
        xmh[2] = xc[2]-h
        gmh = self.constraints(xmh)
        jac[:,2] = (gph-gmh)/(2.0*h)
        
        return jac




#Reference implementation from LANCELOT adopted for PyTorch
class AugmentedLagrangianModel:
    def __init__(self, x10, x1f, x20, x2f):
        self.M = x10.size(0) #Number of samples
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f
        self.factor = 0.5
        self.min_lr = 1.0e-8
        self.mup0 = 10.0
        self.omegap0 = 1.0/self.mup0
        self.etap0 = 1.0/(self.mup0**0.1)
        #Create modest values for the convergence criteria
        self.omegastar = 1.0e-4
        self.etastar = 1.0e-4
        self.lambdap0 = torch.ones((self.M,3), dtype=torch.float64, device=device)
        self.NAdam = 200
        self.mup = torch.ones_like(x10) * self.mup0
        self.omegap = torch.ones_like(x10) * self.omegap0
        self.etap = torch.ones_like(x10) * self.etap0
        self.lambdap = torch.ones_like(x10) * self.lambdap0
        self.maxiter = 1000
        self.maxoptiter = 1000
        self.lr =  0.1#Initial learning rate
        self.saved_state = None  # Attribute to store the in-memory state
        self.xmint = torch.tensor([-0.1, -0.1, 0.01], dtype=dtype, device=device)
        self.xmaxt = torch.tensor([0.1, 0.1, 10.0], dtype=dtype, device=device)


    
    def save_state_in_memory(self, optimizer):
        # Save the model and optimizer state along with any other relevant information in memory
        self.saved_state = {
            'optimizer_state_dict': optimizer.state_dict(),
        }
        print("State saved in memory")

    def schedule(self, optimizer):
        for param_group in optimizer.param_groups:
          new_lr = max(param_group['lr'] * self.factor, self.min_lr)
          param_group['lr'] = new_lr
          print(f"Learning rate reduced to {new_lr:.6f}")
          self.lr = new_lr
        self.save_state_in_memory(optimizer)

    def P(self, x, gradL, l, u):
        """Projection operator."""
        return torch.max(torch.min(x-gradL, u), l)

    def optimize(self, x):
        """Target function to minimize."""
        m = torch.ones((self.M), dtype=torch.bool, device=device)
        xt = x
        #Use the Adam optimizer to minimize the function
        count = 0
        self.x10.requires_grad_(False)
        self.x20.requires_grad_(False)
        self.x1f.requires_grad_(False)
        self.x2f.requires_grad_(False)
        self.lambdap.requires_grad_(False)
        self.mup.requires_grad_(False)
        
        cm = None

        while (m.numel() > 0) and (count < self.maxiter):
          J = torch.ones_like(self.x10)*100.0

          count = count + 1
          #It is necessary to detach the tensors here to avoid going through the same graph twice
          xt.detach_()
          self.x10.detach_()
          self.x20.detach_()
          self.x1f.detach_()
          self.x2f.detach_()
          self.lambdap.detach_()
          self.mup.detach_()

          xtm = xt[m]

          xtm.requires_grad_(True)
          #Set the parameters in such a way so that the optimizer trusts the gradients
          #Because dual numbers are very precise and noise free
          #optimizer = torch.optim.Adam([xtm], betas=(0.8, 0.999), lr=self.lr, amsgrad=True)
          optimizer = torch.optim.Adam([xtm]) 
          if self.saved_state is not None:
            optimizer.load_state_dict(self.saved_state['optimizer_state_dict'])
            print("Optimizer state loaded")
          # Create the ReduceLROnPlateau scheduler
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='min',  # 'min' for minimizing the monitored metric (e.g., validation loss)
                                                 factor=0.1,  # Factor by which the learning rate will be reduced (new_lr = lr * factor)
                                                 patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
                                                 threshold=0.0001,  # Threshold for measuring the new optimum
                                                 threshold_mode='rel',  # 'rel' or 'abs' to compare with the best
                                                 cooldown=0,  # Number of epochs to wait before resuming normal operation after lr has been reduced
                                                 min_lr=0,  # Lower bound on the learning rate
                                                 eps=1e-08)  # Minimal decay applied to the lr, if it is the same as the last lr

          scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-8)
          J, cm = VdpNecCond.apply(xtm, self.x10[m], self.x20[m], self.x1f[m], self.x2f[m], self.lambdap[m], self.mup[m])
          J.backward()
          Pval = self.P(xtm, xtm.grad, self.xmint, self.xmaxt)
          xmPnorm = (xtm-Pval).norm(1,keepdim=True)

          countOpt = 1

          while torch.any(xmPnorm > self.omegap,1):
            countOpt = countOpt + 1
            optimizer.zero_grad()
            J, cm = VdpNecCond.apply(xtm, self.x10[m], self.x20[m], self.x1f[m], self.x2f[m], self.lambdap[m], self.mup[m])
            J.backward()
            Pval = self.P(xtm, xtm.grad, self.xmint, self.xmaxt)
            xmPnorm = (xtm-Pval).norm(1,keepdim=True)
            print(f"J: {J}")
            print(f"xmPnorm for count {countOpt}: {xmPnorm}")
            print(f"cm: {cm}")

            optimizer.step()
          scheduler.step(J)
          #if countOpt == self.NAdam:
          #  print("Optimization failed reducing step size")
          #  self.schedule(optimizer)
          #  #update the original tensor anyway keep any progress made
          #  xt[m] = xtm.detach()
          #  continue

          #update the original tensor
          xt[m] = xtm.detach()
          #Apply the backward pass once
          print(f"J: {J}")
          print(f"cm: {cm}")
          


          #Run the optimizer until we can obtain some degree of convergence
          #around the necessary conditions
          
          #while (xmPm > self.omegap).all() and countopt < self.maxoptiter:
          #  countopt = countopt + 1
            #print(f"cm norm: {cm.norm(1,keepdim=True)}")
            #print(f"xtm.grad.norm(): {xmPm} versus target: {self.omegap}")
            #optimizer.zero_grad()
            #J, cm = VdpNecCond.apply(xtm, self.x10[m], self.x20[m], self.x1f[m], self.x2f[m], self.lambdap[m], self.mup[m])
            #print(f"J: {J}")
            #print(f"cm: {cm}")
            #J.norm(dim=1,keepdim=True).backward()
          #  J = optimizer.step(closure)
          #  print(f"J: {J}")
          #  print(f"cm: {cm}")
          #  print(f"xtm.grad.norm() during while loop: {xmPm}")
          #  xmPm = xtm.grad.norm(1,keepdim=True)
            
          #print(f"xtm after optimization: {xtm}")
          #xt[m] = xtm.detach()
          #There are three different cases
          m1 = torch.all((cm.norm(1,keepdim=True) <= self.etastar), 1) & torch.all((xmPnorm <= self.omegastar), dim=1)
          m2 = ~m1 & torch.all((cm.norm(1,keepdim=True) <= self.etap[m]), dim=1)
          m3 = ~m1 & torch.all((cm.norm(1,keepdim=True) > self.etap[m]), dim=1)
          print(f"m1: {m1}")
          print(f"m2: {m2}")
          print(f"m3: {m3}")
          print(f"cm: {cm}")
          #Expand the masks to the original size
          self.save_state_in_memory(optimizer)

          #Apply the second case to the parameters
          if (torch.any(m2)):
            print("Applying case 2")
            indcs = m.nonzero()
            filtindcs = indcs[m2]
            self.lambdap.index_put_((filtindcs,), self.lambdap[filtindcs]-self.mup[filtindcs]*cm[m2])
            self.etap.index_put_((filtindcs,), self.etap[filtindcs]/self.mup[filtindcs].pow(0.9))
            self.omegap.index_put_((filtindcs,), self.omegap[filtindcs]/self.mup[filtindcs])
          #Apply the third case to the parameters
          if (torch.any(m3)):
            print("Applying case 3")
            print(f"m3: {m3}")
            print(f"m: {m}")
            print(f"mup before: {self.mup}")
            indcs = m.nonzero()
            filtindcs = indcs[m3]
            self.mup.index_put_((filtindcs,), self.mup[filtindcs]*100.0)
            self.etap.index_put_((filtindcs,), self.mup[filtindcs].pow(0.1).reciprocal())
            self.omegap.index_put_((filtindcs,), self.mup[filtindcs].reciprocal())

          #Finally update the mask for those elements that have converged
          if (torch.any(m1)):
            print("Applying case 1")
            indcs = m.nonzero()
            filtindcs = indcs[m1]
            m.index_put_((filtindcs,), False)
          print(f"Before next while loop m: {m}")
          print(f"xt: {xt}")
          print(f"self.lambdap: {self.lambdap}")
          print(f"self.mup: {self.mup}")
          print(f"self.etap: {self.etap}")
          print(f"self.omegap: {self.omegap}")
        return xt.detach()


def batched_objective_function_ipopt(x):
  # Example bounds for three parameters
  # Define parameter bounds-there are not necessarily the same 
  # as the ones used in the outer optimization
  # The job of the outer optimization is to find the best initial guess
  # for the inner optimization so the inner optimization can have wider parameters
  p20min, p20max = -10.0, 10.0
  ftmin, ftmax   = 1.0, 25.0


  mu = 3.0
  W = 1.0
  print(f"Setting up the problem with mu: {mu} and W: {W}")
  print(f"Setting up the problem with initial conditions {x10}, {x1f}, {x20}, {x2f}.")
  M = x.size(0)
  res = torch.zeros((M,1), dtype=torch.float64, device=device)
  for i in range(M):
    problem = VDPMintIpopt(x10, x1f, x20, x2f, mu, W)
    nlp = cyipopt.Problem(
            n=2,
            m=2,
            problem_obj=problem,
            lb=[p20min, ftmin],
            ub=[p20max, ftmax],
            #Loosen the constraints to achieve convergence
            cl=[-1.0e-6, -1.0e-6],
            cu=[1.0e-6,  1.0e-6]
        )

    # Set the options
    nlp.add_option('hessian_approximation', 'limited-memory')  # Enable limited memory BFGS (L-BFGS)
    nlp.add_option('linear_solver', 'mumps')  # Set MUMPS as the linear solver
    nlp.add_option('tol', 1e-4)               # Set the tolerance to 10^-4
    nlp.add_option('print_level', 5)          # Set print level to 5
    nlp.add_option('max_iter', 200)       # Set the maximum number of iterations to 1000
    nlp.add_option('mu_strategy', 'adaptive')  # Set the barrier parameter strategy to adaptive
    nlp.add_option("derivative_test", "first-order")  # Check the gradient
    nlp.add_option("derivative_test_tol", 1.0e-6)  # Set the gradient check tolerance
    
    x0 = x[i].flatten().numpy()
    print(f"Initial guess: {x0}")
    sol, info = nlp.solve(x0)
    print(f"Received info: {info}")
    print(f"Optimal solution: {sol}")
    if info['status'] == 0:
      res[i] = torch.tensor(problem.objective(sol))
      #convert to tensor
    else:
      res[i] = torch.tensor(1000.0)
  return res

#The inputs are all tensors
def batched_augLang_ipopt(xics, x, lambdap, mup, tol):
  # Example bounds for three parameters
  # Define parameter bounds-there are not necessarily the same 
  # as the ones used in the outer optimization
  # The job of the outer optimization is to find the best initial guess
  # for the inner optimization so the inner optimization can have wider parameters

  print(f"Setting up the problem with mu: {mu} and W: {W}")
  print(f"Setting up the problem with initial conditions {xics}.")
  M = x.size(0)
  print(f"Setting up the problem with sample size {M}.")
  
  res     = torch.zeros((M,1), dtype=torch.float64, device=device)
  sols    = torch.zeros((M,3), dtype=torch.float64, device=device)
  grads   = torch.zeros((M,3), dtype=torch.float64, device=device)
  cs      = torch.zeros((M,5), dtype=torch.float64, device=device)
  cnorm   = torch.zeros((M,), dtype=torch.float64, device=device)
  problem = VDPAugPMPIpopt(xics, lambdap, mup, x.flatten().numpy())

  print(f'problem = {problem}')
  xlb = [0.0 for _ in range(3*M)]
  xub = [0.0 for _ in range(3*M)]
  #Limit the exploration space
  for i in range(M):
    if x[i,0] < 0: 
      xlb[3*i]   = x[i,0]*2.0
      xub[3*i]   = x[i,0]*0.5
    else:
      xlb[3*i]   = x[i,0]*0.5
      xub[3*i]   = x[i,0]*2.0
    if x[i, 1] < 0:   
      xlb[3*i+1] = x[i, 1]*2.0
      xub[3*i+1] = x[i, 1]*0.5
    else:
      xlb[3*i+1] = x[i, 1]*0.5
      xub[3*i+1] = x[i, 1]*2.0       
    #Limit the time to avoid long integrations
    xlb[3*i+2] = x[i, 2]*0.5
    xub[3*i+2] = x[i, 2]*2



  x0 = x.flatten().numpy().tolist()
  print(f"Initial guess: {x0}")
  print(f'len(x0): {len(x0)}')
  print(f'len(xlb): {len(xlb)}')
  print(f'len(xub): {len(xub)}')
  params = torch.tensor([problem.rtol, problem.atol], dtype=torch.float64)
  
  [res, grads, cs, cnorm, jac] = \
                              janus_nlp.mint_auglangr_propagate(problem.xics, x, problem.lambdap, problem.mup, params)

  scaling_factor = 1.0/res.abs().item()
  print(f'Problem scaling_factor={scaling_factor}')
  nlp = cyipopt.Problem(
            n=3*M,
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




def batched_objective_function(xt):
    # Example bounds for three parameters
    # Define parameter bounds-there are not necessarily the same 
    # as the ones used in the outer optimization
    # The job of the outer optimization is to find the best initial guess
    # for the inner optimization so the inner optimization can have wider parameters

    rtol = 1.0e-6
    atol = 1.0e-6
    params = torch.tensor([rtol, atol], dtype=torch.float64)
    print(f"xt: {xt}")
    M = xt.size(0)
    objs = torch.zeros((M,1), dtype=torch.float64, device=device)
    c1s = torch.zeros((M,1), dtype=torch.float64, device=device)
    c2s = torch.zeros((M,1), dtype=torch.float64, device=device)
    x10t = torch.tensor([x10], dtype=dtype, device=device)
    x20t = torch.tensor([x20], dtype=dtype, device=device)
    for i in range(M):
       p20 = xt[i, 0].item()
       ft = xt[i, 1].item()
       p10 = janus_nlp.calc_p1(xt[i, 0], x10t, x20t).item()
       LI0 = 0.0
       t_span = (0, ft)
       #Use scipy to propagate the augmeted dynamics
       y0 = [x10, x20, p10, p20, LI0]
       print(f"Initial conditions: {y0}")
       sol = solve_ivp(dyns_aug, t_span, y0, method='Radau', rtol=rtol, atol=atol)
       #The objective 
       #This is just a rough approximation
       c1s[i] = sol.y[2,-1]-x1f
       c2s[i] = sol.y[3,-1]-x2f
       objs[i] = torch.tensor(ft+sol.y[4,-1]).flatten()
       c1s[i] = torch.tensor(c1s[i]).flatten()
       c2s[i] = torch.tensor(c2s[i]).flatten()
    return objs, c1s, c2s



  
def initialize_model(train_X, train_Y):
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def optimize_hyperparameters(batch_size=1, n_iterations=10, n_init=10, tolerance=1e-2):
    dtype = torch.double
    # Define the bounds as a tensor
    p20minB, p20maxB = -1.0, 1.0
    ftminB, ftmaxB   = 0.01, 10.0

    bounds = torch.tensor([
      [p20minB, ftminB],  # Lower bounds
      [p20maxB, ftmaxB]   # Upper bounds
    ], dtype=dtype, device=device)

    # Generate initial points in the original space
    train_X = draw_sobol_samples(bounds=bounds, n=n_init, q=batch_size).squeeze(1)
    print(f"Initial points: {train_X}")
    train_Y, train_C1, train_C2 = batched_objective_function(train_X)
    print(f"Initial values: {train_Y}")
    print(f"Initial constraints 1: {train_C1}")
    print(f"Initial constraints 2: {train_C2}")

    for i in range(n_iterations):
        # Normalize inputs and standardize outputs
        train_X_normalized = normalize(train_X, bounds=bounds)
        train_Y_standardized = standardize(train_Y)
        train_C1_standardized = standardize(train_C1)
        train_C2_standardized = standardize(train_C2)
        print(f"Normalized inputs: {train_X_normalized.shape}")
        print(f"Standardized outputs: {train_Y_standardized.shape}")
        print(f"Standardized constraints 1: {train_C1_standardized.shape}")
        print(f"Standardized constraints 2: {train_C2_standardized.shape}")
        
        # Initialize and fit model
        model, mll = initialize_model(train_X_normalized, train_Y_standardized)
        fit_gpytorch_mll(mll)

        # Normalize constraint GPs as well
        constraint_gp1 = SingleTaskGP(train_X_normalized, train_C1_standardized)  # A GP to model the constraint
        mll_constraint1 = ExactMarginalLogLikelihood(constraint_gp1.likelihood, constraint_gp1)
        fit_gpytorch_mll(mll_constraint1)

        constraint_gp2 = SingleTaskGP(train_X_normalized, train_C2_standardized)  # A GP to model the constraint
        mll_constraint2 = ExactMarginalLogLikelihood(constraint_gp2.likelihood, constraint_gp2)
        fit_gpytorch_mll(mll_constraint2)
       
        # Define constraint functions for equality (close to zero)
        def constraint_1(X: torch.Tensor) -> torch.Tensor:
          junk1 = constraint_gp1.posterior(X)
          print(f"junk1={junk1}")
          exit(1)
          return X.mean()-tolerance

        def constraint_2(X: torch.Tensor) -> torch.Tensor:
          junk2 = constraint_gp2.posterior(X)
          print(f"junk2={junk2}")
          exit(1)
          return X.mean() - tolerance

        # Define the constrained acquisition objective
        def objective(samples: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return samples  # The objective is maximized

        
        # Define the acquisition function
        UCB = qUpperConfidenceBound(model, beta=0.1, objective=constrained_objective)
        
        # Optimize the acquisition function in the normalized space
        new_points_normalized, _ = optimize_acqf(
            UCB, 
            bounds=torch.tensor([[0, 0], [1.0, 1.0]], dtype=dtype),
            q=batch_size,
            num_restarts=5,
            raw_samples=10,
        )
        
        # Unnormalize the new points for objective function evaluation
        new_points = unnormalize(new_points_normalized, bounds=bounds)
        print(f"New points shape: {new_points.shape}")
        exit(1)
        
        # Evaluate the objective and constraint functions at the new points
        new_Y, new_C1, new_C2 = batched_objective_function(new_points)
        print(f"New values: {new_Y}")
        print(f"New constraints 1: {new_C1}")
        print(f"New constraints 2: {new_C2}")

        # Update training data
        train_X = torch.cat([train_X, new_points])
        train_Y = torch.cat([train_Y, new_Y])
        train_C1 = torch.cat([train_C1, new_C1])
        train_C2 = torch.cat([train_C2, new_C2])
        
        print(f"Iteration {i+1}: Best values = {train_Y.max(dim=0).values.squeeze().tolist()}")
    
    best_indices = train_Y.argmin(dim=0).squeeze()  # Find the indices of the minimum values in train_Y
    best_hyperparameters = train_X[best_indices, :]  # Select the hyperparameters corresponding to the best values
    best_values = train_Y[best_indices, :]  # Ensure you're getting the actual best values

    return best_hyperparameters, best_values

def dyns_jacobian(t, y):
    p1 = y[0]
    p2 = y[1]
    x1 = y[2]
    x2 = y[3]
    if p1 < 0.0:
        u1star = u1max
    else:
        u1star = u1min
    if p2*(1-x1*x1)*x2 < 0.0:
        u2star = u2max
    else:
        u2star = u2min
    if p2 < 0.0:
         u3star = u3max
    else:
         u3star = u3min      

    jac = np.zeros((4,4))
    #p2 * u2star * (-2 * x1) * x2 - p2;
    #jac.index_put_({Slice(), Slice(0, 1), 1}, u2star * (-2 * x1) * x2 - one);
    jac[0,1]= u2star * (-2 * x1) * x2 - 1.0
    #jac.index_put_({Slice(), Slice(0, 1), 2}, -p2 * u2star * 2 * x2);
    jac[0,2]= -p2 * u2star * 2 * x2
    #jac.index_put_({Slice(), Slice(0, 1), 3}, -p2 * u2star * 2 * x1);
    jac[0,3]= -p2 * u2star * 2 * x1
    #p1 + p2 * u2star * (1 - x1 * x1);
    #jac.index_put_({Slice(), Slice(1, 2), 0}, one);
    jac[1,0]= 1.0
    #jac.index_put_({Slice(), Slice(1, 2), 1}, u2star * (1 - x1 * x1));
    jac[1,1]= u2star * (1 - x1 * x1)
    #jac.index_put_({Slice(), Slice(1, 2), 2}, p2 * u2star * (-2 * x1));
    jac[1,2]= p2 * u2star * (-2 * x1)
    #x2+u1star;
    #jac.index_put_({Slice(), Slice(2, 3), 3}, one);
    jac[2,3]= 1.0
    #u2star*((1-x1*x1)*x2)-x1+u3star
    #jac.index_put_({Slice(), Slice(3, 4), 2}, u2star * (-2 * x1 * x2) - one);
    jac[3,2]= u2star * (-2 * x1 * x2) - 1.0
    #jac.index_put_({Slice(), Slice(3, 4), 3}, u2star * ((1 - x1 * x1)));
    jac[3,3]= u2star * ((1 - x1 * x1))
    return jac


def dyns_aug(t, y):
    p1 = y[0]
    p2 = y[1]
    x1 = y[2]
    x2 = y[3]
    if p1 < 0.0:
        u1star = u1max
    else:
        u1star = u1min
    if p2*(1-x1*x1)*x2 < 0.0:
        u2star = u2max
    else:
        u2star = u2min
    if p2 < 0.0:
         u3star = u3max
    else:
         u3star = u3min      
    dp1dt = p2 * u2star * (-2.0 * x1) * x2 - p2
    dp2dt = p1 + p2 * u2star * (1.0 - x1 * x1)
    dx1dt = x2+u1star
    dx2dt = u2star * (1.0 - x1 * x1) * x2 - x1 + u3star
    return np.stack([dp1dt, dp2dt, dx1dt, dx2dt], axis=-1)

def augmented_objective_function(p10, p20, ft, lambdap1, lambdap2, mup, x10, x20):
    t_span = (0, ft)
    y0 = [p10, p20, x10, x20]
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau', jac = dyns_jacobian ,rtol=1e-9, atol=1e-12)
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
def initialize_smac_with_initial_conditions(scenario, initial_condition):

    # Adjust your objective function if necessary to incorporate initial conditions and seeds
    def optimize_hyperparameters_smac(config, seed=0):
      p1 = config["p1"]
      p2 = config["p2"]
      ft = config["ft"]
      lambdap1 = config["lambdap1"]
      lambdap2 = config["lambdap2"]
      mup = config["mup"]
      x10 = initial_condition[0]
      x20 = initial_condition[1]
      obj= augmented_objective_function(p1, p2, ft, lambdap1, lambdap2, mup, x10, x20)
      print(f"Optimizing with p1={p1}, p2={p2} and ft={ft}")
      print(f"Objective function value: {obj}")
      return obj


    # SMAC facade using your objective function
    smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=optimize_hyperparameters_smac, overwrite=True)

    # Optimize hyperparameters
    return smac

@ray.remote
def do_optimize(initial_conditions):
  lambdap = np.asarray([0.0, 0.0]).astype(np.float64)
  mup = 0.01

  count = 0
  p1 = 0.0
  p2 = 0.0
  ft = (ftmin+ftmax)/2
  ########################################################
  #Penalty method with Global Bayesian Optimization 
  ########################################################
  converged = False

  # Define the configuration space
  cs = ConfigurationSpace(name="vpd config space", space={"p1": Float("p1", bounds=(p10min, p10max), default=p1),
                                                                "p2": Float("p2", bounds=(p20min, p20max), default=p2),
                                                                "ft": Float("ft", bounds=(ftmin, ftmax), default=ft),
                                                                "lambdap1": Constant("lambdap1", value=lambdap[0]),
                                                                "lambdap2": Constant("lambdap2", value=lambdap[1]),
                                                                "mup": Constant("mup", value=mup) })
    
  # Run SMAC to optimize the objective function
  scenario = Scenario(cs, deterministic=False, n_trials=1000)
  optimizer = initialize_smac_with_initial_conditions(scenario, initial_conditions)

  while count < 5:
    count = count + 1
    incumbent = optimizer.optimize()
    print(f"Optimal hyperparameters: {incumbent}")
    p1 = incumbent["p1"]
    p2 = incumbent["p2"]
    ft = incumbent["ft"]
    t_span = (0, ft)
    y0 = [p1, p2, initial_conditions[0], initial_conditions[1]]
    sol = solve_ivp(dyns_aug, t_span, y0, method='Radau',rtol=1e-6, atol=1e-9)
    x1fp = sol.y[2,-1]
    x2fp = sol.y[3,-1]
    cs = np.asarray([x1fp+np.abs(x1fp), x2fp-x2f])
    cnorms = np.linalg.norm([x1fp+np.abs(x1fp), x2fp-x2f])
    if cnorms < 1e-3:
      print(f"cnorms converged to {cnorms} count {count} for initial conditions: {initial_conditions}")
      converged = True
      break
    else:
      converged = False
      print(f"Applying case 2 cnorms: {cnorms} count {count} initial conditions: {initial_conditions}")
      mup=mup*100.0
      
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
    ics = torch.tensor(initial_conditions, dtype=torch.float64, device=device).reshape(1, 2)
    xopt = torch.tensor([[p1, p2, ft]], dtype=torch.float64, device=device)
    print(f"Initial guess: {xopt}")
    print(f"Initial lambdap: {lambdap}")
    print(f"Initial mup: {mup}")
    print(f"Initial omegap: {omegap}")
    print(f"Initial etap: {etap}")
    
    while (cnorms > 1.0e-6).any() and count < 5:
      omegap, xopt, grads, cs, cnorms, jac= batched_augLang_ipopt(ics, xopt, lambdap, mup, omegap.mean())
      if (omegap < 1.0e-9 and cnorms < 1.0e-6 ).all():
        print(f'Finished optimization')
        break

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
    #Update the values
    p1 = xopt[0,0].item()
    p2 = xopt[0,1].item()
    ft = xopt[0,2].item()
  else:
     print(f'Failed to converge skipping full ALM')
  return p1, p2, ft, converged

def augmented_opt(iteration=0, numSamples=2):
    # Define bounds for the 2D Sobol sequence
    lower_bounds = torch.tensor([-0.5, 1.0], device=device, dtype=torch.float64)
    upper_bounds = torch.tensor([0.5, 20.0], device=device, dtype=torch.float64)
    sobol = torch.quasirandom.SobolEngine(dimension=2)
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
  ics, phat = augmented_opt(6, 100)
  ray.shutdown()
  print(f"Initial conditions: {ics}")
  print(f"BO results: {phat}")
  #Check the answer by forward propagation
  #We should filter out any trajectories that are not optimal
  #Go through the results and filter out the ones that do not meet the constraints
  ics_opt = torch.zeros((0,2), dtype=torch.float64, device=device)
  phat_opt = torch.zeros((0,3), dtype=torch.float64, device=device)
  #Also record the failed samples
  ics_opt_failed = torch.zeros((0,2), dtype=torch.float64, device=device)
  phat_opt_failed = torch.zeros((0,3), dtype=torch.float64, device=device)

  for i in range(ics.size(0)):
    p1 = phat[i,0]
    p2 = phat[i,1]
    ft = phat[i,2]
    success = phat[i,3]
    if success == 1:
      t_span = (0, ft)
      y0 = [p1, p2, ics[i,0], ics[i,1]]
      print(f"Initial conditions using solve_ivp: {y0}")
      sol = solve_ivp(dyns_aug, t_span, y0, method='Radau',jac=dyns_jacobian, rtol=1e-9, atol=1e-12)
      p1fp = sol.y[0,-1]
      p2fp = sol.y[1,-1]
      x1fp = sol.y[2,-1]
      x2fp = sol.y[3,-1]
      print(f"x1fp from solve_ivp: {x1fp}")
      print(f"x2fp from solve_ivp: {x2fp}")
      print(f"p1fp from solve_ivp: {p1fp}")
      print(f"p2fp from solve_ivp: {p2fp}")
      obj = augmented_objective_function(p1, p2, ft, 0.0, 0.0, 10.0, ics[i][0], ics[i][1])
      print(f"Objective function value: {obj}")
      ics_opt = torch.cat((ics_opt, ics[i].reshape(1,2)))
      phat_opt = torch.cat((phat_opt, phat[i,0:3].reshape(1,3)))
    else:
      ics_opt_failed = torch.cat((ics_opt_failed, ics[i,:].reshape(1,2)))
      phat_opt_failed = torch.cat((phat_opt_failed, phat[i,0:3].reshape(1,3)))
 

  #Now save the optimal initial conditions and the optimal parameters in a file using pickle
  #in double precision
  print(f"Number of succesful solutions {ics_opt.shape[0]}")
  print(f"Number of failed solutions {ics_opt_failed.shape[0]}")
  with open('data/optimal_initial_conditions.pkl', 'ab') as f:
    pickle.dump(ics_opt, f)
  with open('data/optimal_parameters.pkl', 'ab') as f:
    pickle.dump(phat_opt, f)
  with open('data/failed_initial_conditions.pkl', 'ab') as f:
    pickle.dump(ics_opt_failed, f)
  with open('data/failed_parameters.pkl', 'ab') as f:
    pickle.dump(phat_opt_failed, f)
  
