#ifndef newt_HPP_INCLUDED
#define newt_HPP_INCLUDED
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <janus/lute.hpp>
#include <janus/qrte.hpp>
#include "lnsrchte.hpp"
#include <janus/janus_util.hpp>
namespace janus
{

    std::tuple<torch::Tensor, torch::Tensor> newtTe(torch::Tensor &x,
                                                  const torch::Tensor &params,
                                                  const torch::Tensor &xmin,
                                                  const torch::Tensor &xmax,
                                                  const std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)> &func,
                                                  const std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)> &jacfunc)
    {
        int MAXITS = 500;
        //Evaluate the function and the jacobian as well as the internal quadratic objective function J
        auto f = func(x, params);
        auto jac  = jacfunc(x, params);
        auto J = Jfunc(f);
        const auto STPMX = torch::tensor({100.0}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLF = torch::tensor({1.0e-8}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLMIN = torch::tensor({1.0e-12}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLX = torch::tensor(std::numeric_limits<double>::epsilon(), torch::dtype(torch::kFloat64)).to(x.device());
        auto true_t = torch::tensor({true}, torch::dtype(torch::kBool)).to(x.device());
        auto false_t = torch::tensor({false}, torch::dtype(torch::kBool)).to(x.device());
        int M = x.size(0); // Batch size
        int N = x.size(1); // State space size
        auto xnorm = x.square().sum(1).sqrt();
        auto xcomp = torch::ones_like(xnorm)*N;
        auto stpmax = STPMX * bmax(xnorm, xcomp);
        // Is the initial guess a root?
        auto test = std::get<0>(f.abs().max(1, false)); // This is dimension M
        auto check = torch::zeros({M}, torch::dtype(torch::kBool)).to(x.device()); 
        auto xold = x.clone();
        auto fold = f.clone();
        auto Jold = J.clone();
        auto p = torch::zeros_like(f);
        auto g = torch::zeros_like(x);
        int count = 0;
        auto m1 =test < 0.01 * TOLF;   // Do we already have some convergence?
        if ( m1.eq(true_t).any().item<bool>())
        {
            check.index_put_({m1}, false);
        }
        auto m2 = ~m1;  // For readability create a new mask containing the samples that have not converged
        while ( m2.eq(true_t).any().item<bool>() && count < MAXITS)
        {
            count++;
            std::cerr << "Iteration count = " << count << std::endl;
            //Recalculate the function and the jacobian
            f.index_put_({m2}, func(x.index({m2}).contiguous(),
                                    params.index({m2}).contiguous()));
            jac.index_put_({m2}, jacfunc(x.index({m2}).contiguous(),
                                 params.index({m2}).contiguous()));
            //Calculate the gradient
            g.index_put_({m2}, torch::einsum("mij, mi->mj", {jac.index({m2}).contiguous(),
                                                             f.index({m2}).contiguous()}));
            xold.index_put_({m2}, x.index({m2}).contiguous());
            fold.index_put_({m2}, f.index({m2}).contiguous());
            Jold.index_put_({m2}, J.index({m2}).contiguous());
            // Since the jacobian has been recalcualted, we need to recompute the QR decomposition
            //auto [qt, r] = qrte(jac.index({check}).contiguous()); // This needs to be replaced with GMRES for vey high dimensional systems
            auto jacm2 = jac.index({m2}).contiguous();
            //auto [qt, r] = qrte(jacm2);
            auto [LU, P] = LUTe(jacm2);
            auto pm2 = -f.index({m2}).contiguous();
            auto sollu = solveluv(LU, P, pm2);
            //auto sol = qrtesolvev(qt, r, pm2);
            //Test the solution
            //auto Jacsollu = torch::einsum("mij, mj->mi", {jacm2, sollu});
            //std::cerr << "Error from LU decomposition=" << (Jacsollu - pm2).norm(2, 1) << std::endl;
            //auto Jacsolqr = torch::einsum("mij, mj->mi", {jacm2, sol});
            //std::cerr << "Error from QR decomposition=" << (Jacsolqr - pm2).norm(2, 1) << std::endl;
            p.index_put_({m2}, sollu);
            // Update x, p and f
            auto xoldin = xold.index({m2}).contiguous();
            auto foldin = fold.index({m2}).contiguous();
            auto joldin = Jold.index({m2}).contiguous();
            auto gin = g.index({m2}).contiguous();
            auto pin = p.index({m2}).contiguous();
            auto stpmaxin = stpmax.index({m2}).contiguous();
            auto paramsin = params.index({m2}).contiguous();
            std::cerr << "At count = " << count << std::endl;
            std::cerr << "Input into lnsrch " << xoldin << std::endl;
            auto [xs, js, ps, checkupd] = lnsrchTe(xoldin,
                                                   foldin,
                                                   joldin,
                                                   gin,
                                                   pin,
                                                   stpmaxin,
                                                   paramsin,
                                                   xmin.index({m2}),
                                                   xmax.index({m2}),
                                                   func);
            std::cerr << "Output from lnsrch " << xs << std::endl;
            std::cerr << "Error at count=" << count << " "<< js << std::endl;
            std::cerr << "checkupd at count=" << count << " "<< checkupd << std::endl;
            x.index_put_({m2}, xs);
            p.index_put_({m2}, ps);
            J.index_put_({m2}, js);
            check.index_put_({m2}, checkupd);

            test.index_put_({m2}, std::get<0>(f.index({m2}).abs().max(1, false)));
            auto m2_1 = m2 & (test < TOLF);
            if (m2_1.any().eq(true_t).item<bool>())
            {
              check.index_put_({m2_1}, false);
              m2.index_put_({m2_1}, false);
            }
            

            // Need to recheck the flags since they have been updated
            auto m2_2 = m2 & check;
            if ((m2_2).any().equal(true_t))
            {
                //Checks for convergence on the gradient
                test.index_put_({m2_2}, 0.0);
                auto Jl = J.index({m2_2}).contiguous();
                auto den = torch::max(Jl, 0.5 * torch::ones_like(Jl) * N);
                auto xt = x.index({m2_2}).abs();
                auto ONE = torch::ones_like(xt);
                auto rhs = torch::einsum("mi,m->mi", {bmax(xt, ONE),  den.reciprocal()});
                auto temp = torch::einsum("mi,mi->m",{g.index({m2_2}).abs() ,
                                                      rhs});
                test.index_put_({m2_2}, std::get<0>(temp.max(1, false)));
                check.index_put_({m2_2}, test.index({m2_2}) < TOLMIN);
                m2.index_put_( {m2_2}, false); //We are done with this samples 

            }
            //Recheck for the check flags since they have been updated
            if (m2.eq(true_t).any().item<bool>())
            {
                // Check for small relative changes in x
                auto dx = (x.index({m2}) - xold.index({m2})).abs();
                auto xabs = x.index({m2}).abs();
                test.index_put_({m2}, std::get<0>((dx/xabs).max(1)));
                if ( (test.index({m2}) < TOLX).any().item<bool>() )
                {
                    std::cerr << "Small relative changes in x detected" << std::endl;
                }

                m2.index_put_({m2.clone()}, ~(test.index({m2}) < TOLX));
            }
        }
        
        return std::make_tuple(x, check);
    }

    
} // namespace janus

#endif // newt_HPP_INCLUDED