from newt import newt
from newt_batch import newt_batch
from newt_dual import newt_dual
import torch
from tensordual import TensorDual
from tensormatdual import TensorMatDual
# A simple quadratic function
def simple_quad(x):
    return torch.tensor([(x[0]-1.0)**2, (x[1]-2.0)**2])

# The Jacobian of the simple quadratic function
def simple_quad_jac(x):
    return torch.tensor([[2*(x[0]-1.0), 0], [0, 2*(x[1]-2.0)]])


x_root, check = newt(torch.tensor([100.0, 10.0]), simple_quad, simple_quad_jac)
print("Root: ", x_root)
print("Check: ", check)


def batch_quad(x):
    M, D = x.shape
    res = torch.zeros((M, D) , dtype=x.dtype, device=x.device)
    res[:,0] =  (x[:, 0]-1.0)**2
    res[:,1] =  (x[:, 1]-2.0)**2
    return res

# The Jacobian of the simple quadratic function
def batch_quad_jac(x):
    M, D = x.shape
    res = torch.zeros((M, D, D), dtype=x.dtype, device=x.device)
    res[:, 0,  0] = 2*(x[:,0]-1.0)
    res[:, 0, 1] = 0.0
    res[:, 1, 0] = 0.0
    res[:, 1, 1] = 2*(x[:,1]-2.0)
    return res


def dual_quad(x):
    res = TensorDual.zeros_like(x)
    res[0] =  (x[0]-1.0)**2
    res[1] =  (x[1]-2.0)**2
    return res

# The Jacobian of the simple quadratic function
def dual_quad_jac(x):
    #Create an zero TensorMatDual using x as a vector
    res = TensorMatDual.createZeroJacFromVec(x)
    res[:, 0, 0] = 2*(x[0]-1.0)
    res[:, 0, 1]  = 0.0
    res[:, 1, 0]  = 0.0
    res[:, 1, 1]  = 2*(x[1]-2.0)
    return res

xis=torch.tensor([[200, 30.0], [150., 100.]], dtype=torch.float64)
print(xis[:,0])
x_root, check = newt_batch(xis, batch_quad, batch_quad_jac)

xisd=TensorDual.create(xis)

x_rootd, checkd = newt_dual(xisd, dual_quad, dual_quad_jac)

print("Roots: ", x_root)
print("Checks: ", check)
