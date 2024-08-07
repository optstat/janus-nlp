import numpy as np
import torch
import matplotlib.pyplot as plt
import janus_nlp

# Define parameter bounds
p20min, p20max = -10.0, 10.0
ftmin, ftmax = 0.01, 10.0
x1f, x2f = 1.0, -1.0
n = 1
device = torch.device("cpu")
dtype = torch.double

p20 = torch.tensor([[0.0]]).to(device=device, dtype=dtype)
ft = torch.tensor([[1.0]]).to(device=device, dtype=dtype)
x1ft = torch.ones_like(p20) * x1f
x2ft = torch.ones_like(p20) * x2f
janus_nlp.set_xf(x1ft, x2ft)
p10 = janus_nlp.calc_p10(p20)
print(p10)
print(p20)
print(ft)
traj = janus_nlp.vdp_solve_traj(p10, p20, ft)

print(traj)

