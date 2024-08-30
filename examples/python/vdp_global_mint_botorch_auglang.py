import torch
import janus_nlp
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import gradcheck

#Implementation of the Augmented Lagrangian function using PyTorch
#and Adam as the optimizer

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
W = 0.1

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
        janus_nlp.set_auglangr_W(W)
        #            std::tuple<torch::Tensor, torch::Tensor> mint_auglangr_propagate(const torch::Tensor &x,
        #                                                                     const torch::Tensor &lambdap,
        #                                                                     const torch::Tensor &mup,           
        #                                                                     const torch::Tensor &params)
        params = torch.tensor([1.0e-9, 1.0e-12], dtype=torch.float64)
        output, cm, grads = janus_nlp.mint_auglangr_propagate(input, lambdap, mup, params)
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
    

#Reference implementation from LANCELOT adopted for PyTorch
class AugmentedLagrangianModel:
    def __init__(self, x10, x1f, x20, x2f):
        self.M = x10.size(0) #Number of samples
        self.x10 = x10
        self.x1f = x1f
        self.x20 = x20
        self.x2f = x2f
        self.mup0 = 10.0
        self.omegap0 = 1.0/self.mup0
        self.etap0 = 1.0/(self.mup0**0.1)
        self.omegastar = 1.0e-6
        self.etastar = 1.0e-6
        self.lambdap0 = torch.ones((self.M,2), dtype=torch.float64, device=device)
        self.NAdam = 200
        self.mup = torch.ones_like(x10) * self.mup0
        self.omegap = torch.ones_like(x10) * self.omegap0
        self.etap = torch.ones_like(x10) * self.etap0
        self.lambdap = torch.ones_like(x10) * self.lambdap0
        self.maxiter = 1000
        self.maxoptiter = 1000
        self.lr = 0.001

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
          J = torch.ones_like(self.x10)*100.0#Start a new computational graph

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
    
          optimizer = torch.optim.AdamW([xtm], lr=self.lr)
          optimizer.zero_grad()
          countOpt = 0
          J, cm = VdpNecCond.apply(xtm, self.x10[m], self.x20[m], self.x1f[m], self.x2f[m], self.lambdap[m], self.mup[m])
          J.backward()
          xmPm = xtm.grad.norm(1,keepdim=True)
          optimizer.step()


          while torch.any(xmPm > self.omegap,1):
            print(f"J: {J}")
            print(f"xtm.grad.norm() for count {countOpt}: {xmPm}")
            countOpt = countOpt + 1
            optimizer.zero_grad()
            J, cm = VdpNecCond.apply(xtm, self.x10[m], self.x20[m], self.x1f[m], self.x2f[m], self.lambdap[m], self.mup[m])
            J.backward()
            xmPm = xtm.grad.norm(1,keepdim=True)
            optimizer.step()

          #update the original tensor
          xt[m] = xtm.detach()
          #Apply the backward pass once
          print(f"J: {J}")
          print(f"cm: {cm}")
          


          #Run the optimizer until we can obtain some degree of convergence
          #around the necessary conditions
          countopt = 0
          xmPm = xtm.grad.norm(1,keepdim=True)
          print(f"xtm.grad.norm() before optimization: {xmPm}")
          print("etap: ", self.etap)
          
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
          m1 = torch.all((cm.norm(1,keepdim=True) <= self.etastar), 1) & torch.all((xmPm <= self.omegastar), dim=1)
          m2 = torch.all(~m1 & (cm.norm(1,keepdim=True) <= self.etap[m]), dim=1)
          m3 = torch.all(~m1 & (cm.norm(1,keepdim=True) > self.etap[m]), dim=1)
          #Expand the masks to the original size

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
            self.etap.index_put_((filtindcs,), self.mup[filtindcs].pow(-0.1))
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




if __name__ == "__main__":
  # Example bounds for three parameters
  p20min, p20max = -0.1, 0.1
  ftmin, ftmax   = 1.0, 10.0
  x1f = -1.0
  x2f = -2.0
  x10 = 0.0
  x20 = 6.0
  M = 1 # Number of samples

  # Define the bounds as a tensor
  bounds = torch.tensor([
    [p20min, ftmin],  # Lower bounds
    [p20max, ftmax]   # Upper bounds
  ], dtype=torch.float64, device=device)

  #Generate a Sobol sequence between the bounds
  n_samples = 1
  # Create a Sobol sequence generator
  sobol = qmc.Sobol(d=2, scramble=False) #This retains the sample dimension at 0

  # Generate 1 samples
  n = 1
  samples = sobol.random(n)
  # Rescale the samples to the desired bounds
  lower_bounds = bounds[0]
  upper_bounds = bounds[1]

  # The scaling formula is:
  # scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * samples
  X_train = lower_bounds + (upper_bounds - lower_bounds) * samples




  print(f'X_train shape: {X_train.shape}')
  x1ft = torch.ones((M,1), dtype=torch.float64, device=device) * x1f
  x2ft = torch.ones((M,1), dtype=torch.float64, device=device) * x2f
  x10t = torch.ones((M,1), dtype=torch.float64, device=device) * x10
  x20t = torch.ones((M,1), dtype=torch.float64, device=device) * x20
  model = AugmentedLagrangianModel(x10t, x1ft, x20t, x2ft)
  Y_train = model.optimize(X_train)
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
