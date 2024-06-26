import torch
from linesrch import linesrch
from LU import LU

class Funmin:
    def __init__(self, func):
        self.func = func
        self.fvec = None
        self.n = 0

    def __call__(self, x):
        self.n = x.shape[0]
        self.fvec = self.func(x)
        sum_val = torch.sum(self.fvec ** 2)
        return self.fvec, 0.5 * sum_val

def newt(x, vecfunc, jacfunc):
    MAXITS = 200
    TOLF = 1.0e-8
    TOLMIN = 1.0e-12
    STPMX = 100.0
    TOLX = torch.finfo(torch.float64).eps
    n = x.shape[0]
    g = torch.zeros(n)
    fmin = Funmin(vecfunc)
    fvec, f = fmin(x)
    test = torch.max(torch.abs(fvec))

    if test < 0.01*TOLF:
        check = False
        return x, check
    sum_val = torch.sum(x**2)
    stpmax = STPMX * max(torch.sqrt(sum_val), n)
    for its in range(MAXITS):
        fjac = jacfunc(x)
        fvec, f = fmin(x)
        for i in range(n):
            sum_val = torch.sum(fjac[:,i] * fvec)
            g[i] = sum_val
        xold = x.clone()
        fold = f
        p   = -fvec.clone()
        alu = LU(fjac)
        p   = alu.solvev(p)
        x, f, check= linesrch(xold, fold, g, p, stpmax, fmin)
        test = torch.max(torch.abs(fvec))
        if test < TOLF:
            check = False
            return x, check
        if check:
            test = 0.0
            den = max(f, 0.5*n)
            for i in range(n):
                temp = torch.abs(g[i])*max(torch.abs(x[i]), 1.0)/den
                if temp > test:
                    test = temp
            check = (test < TOLMIN)
            return x, check
        test = 0.0
        for i in range(n):
            temp = torch.abs((x[i] - xold[i]))/max(torch.abs(x[i]),1.0)
            if temp > test:
                test = temp
        if test < TOLX:
            return x, check
    raise Exception("MAXITS exceeded in newt")
