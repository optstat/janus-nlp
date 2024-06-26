import torch
from linesrch_batch import linesrch_batch
from LUBatch import LUBatch

class Funmin_batch:
    def __init__(self, func):
        self.func = func
        self.fvec = None

    def __call__(self, x):
        self.fvec = self.func(x)
        sum_val = torch.sum(self.fvec ** 2, dim=-1, keepdim=True)
        return self.fvec, 0.5 * sum_val


def newt_batch(x, vecfunc, jacfunc):
    MAXITS = 200
    TOLF = torch.tensor(1.0e-8, dtype=x.dtype, device=x.device)
    TOLMIN = torch.tensor(1.0e-12, dtype=x.dtype, device=x.device)
    STPMX = torch.tensor(100.0, dtype=x.dtype, device=x.device)
    TOLX = torch.tensor(torch.finfo(x.dtype).eps, dtype=x.dtype, device=x.device)
    M, n = x.shape
    g = torch.zeros((M, n), dtype=x.dtype, device=x.device)
    fmin = Funmin_batch(vecfunc)
    fvec, f = fmin(x)
    test = torch.max(torch.abs(fvec), dim=-1, keepdim=True).values
    check = test < 0.01 * TOLF
    if torch.all(check):
        return x, check

    sum_val = torch.sum(x ** 2, dim=-1, keepdim=True)
    stpmax = STPMX * torch.where(torch.sqrt(sum_val) > n, torch.sqrt(sum_val),
                                 torch.tensor(n, dtype=x.dtype, device=x.device))
    for its in range(MAXITS):
        fjac = jacfunc(x)
        fvec, f = fmin(x)
        g = torch.einsum('ijk,ik->ij', fjac, fvec)  # batch dot product
        xold = x.clone()
        fold = f
        p = -fvec.clone()
        alu = LUBatch(fjac)
        p = alu.solvev(p)
        x, f, check = linesrch_batch(xold, fold, g, p, stpmax, fmin)
        test = torch.max(torch.abs(fvec), dim=-1, keepdim=True).values
        if torch.all(test < TOLF):
            check = torch.full_like(test, False, dtype=torch.bool)
            return x, check
        if torch.any(check):
            test = 0.0
            den = torch.max(f, 0.5 * n)
            temp = torch.abs(g) * torch.max(torch.abs(x), torch.tensor(1.0, dtype=x.dtype, device=x.device), dim=-1, keepdim=True).values / den
            test = torch.max(temp, dim=-1, keepdim=True).values
            check = test < TOLMIN
            return x, check
        test = torch.zeros((M, 1), dtype=x.dtype, device=x.device)
        for i in range(n):
            max_x = torch.max(torch.abs(x[:, i:i + 1]), torch.tensor(1.0, dtype=x.dtype, device=x.device))
            max_x = max_x.expand_as(x[:, i:i + 1])
            temp = torch.abs((x[:, i:i + 1] - xold[:, i:i + 1])) / max_x
            test = torch.where(temp > test, temp, test)
        check = (test < TOLX.to(x.device)).view(-1)

        if torch.all(test < TOLX):
            return x, check
    raise Exception("MAXITS exceeded in newt")
