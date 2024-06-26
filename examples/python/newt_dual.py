import torch
from linesrch_dual import linesrch_dual
from LUDual import LUDual
from tensordual import TensorDual
from tensormatdual import TensorMatDual
class Funmin_dual:
    def __init__(self, func):
        self.func = func
        self.fvec = None

    def __call__(self, x):
        self.fvec = self.func(x)
        sum_val = TensorDual.sum(self.fvec ** 2)
        return self.fvec, 0.5 * sum_val


def newt_dual(x, vecfunc, jacfunc):
    MAXITS = 200
    TOLF = TensorDual.ones_like(x[0])*1.0e-8
    TOLMIN = TensorDual.ones_like(x[0])*1.0e-12
    STPMX = TensorDual.ones_like(x[0])*100.0
    TOLX = TensorDual.ones_like(x[0])*torch.finfo(x.r.dtype).eps
    M, n = x.r.shape
    g = TensorDual.zeros_like(x)
    fmin = Funmin_dual(vecfunc)
    fvec, f = fmin(x)
    test = TensorDual.max(TensorDual.abs(fvec))
    check = test < 0.01 * TOLF
    if TensorDual.all(check):
        return x, check

    sum_val = TensorDual.sum(x ** 2)
    rootsum_val = TensorDual.sqrt(sum_val)
    stpmax = STPMX * TensorDual.where(TensorDual.sqrt(sum_val) > n, rootsum_val, TensorDual.ones_like(rootsum_val) * n)
    for its in range(MAXITS):
        fjac = jacfunc(x)
        fvec, f = fmin(x)
        g = fjac*fvec  # batch dot product
        xold = x.clone()
        fold = f
        p = -fvec.clone()
        alu = LUDual(fjac)
        p = alu.solvev(p)
        x, f, check = linesrch_dual(xold, fold, g, p, stpmax, fmin)
        test = TensorDual.max(TensorDual.abs(fvec))
        if TensorDual.all(test < TOLF):
            check = TensorDual.full_like(test, False, dtype=torch.bool)
            return x, check
        if TensorDual.any(check):
            den = TensorDual.max(f, 0.5 * n)
            temp = TensorDual.abs(g) * TensorDual.max(TensorDual.abs(x), TensorDual.ones_like(x[0])) / den
            test = TensorDual.max(temp)
            check = test < TOLMIN
            return x, check
        test = TensorDual.zeros_like(x[0])
        for i in range(n):
            arg1 = TensorDual.abs(x[:, i:i + 1])
            arg2 = TensorDual.ones_like(x[0])
            max_x = TensorDual.max_dual(arg1, arg2)
            temp = TensorDual.abs((x[:, i:i + 1] - xold[:, i:i + 1])) / max_x
            test = TensorDual.where(temp > test, temp, test)
        check = (test < TOLX)

        if TensorDual.all(test < TOLX):
            return x, check
    raise Exception("MAXITS exceeded in newt")
