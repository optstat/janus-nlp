#ifndef newted_HPP_INCLUDED
#define newted_HPP_INCLUDED
#include <torch/torch.h>
#include <iostream>
#include <chrono>
//#include <janus/luted.hpp>
#include <janus/qrted.hpp>
#include "lnsrchted.hpp"
#include <janus/janus_util.hpp>
namespace janus
{
    namespace nlp 
    {

    std::tuple<TensorDual, torch::Tensor> newtTeD(TensorDual &x,
                                                  const TensorDual &params,
                                                  const std::function<TensorDual(const TensorDual &, const TensorDual &)> &func,
                                                  const std::function<TensorMatDual(const TensorDual &, const TensorDual &)> &jacfunc)
    {
        int MAXITS = 200;
        auto true_t = torch::tensor({true}, torch::dtype(torch::kBool)).to(x.device());
        auto false_t = torch::tensor({false}, torch::dtype(torch::kBool)).to(x.device());

        auto xold = x.clone();

        auto f = func(x, params);
        auto jac  = jacfunc(x, params);
        auto J = Jfunc(f);

        auto fold = f.clone();
        auto Jold = J.clone();

        const auto STPMX = torch::tensor({100.0}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLF = torch::tensor({1.0e-8}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLMIN = torch::tensor({1.0e-12}, torch::dtype(torch::kFloat64)).to(x.device());
        const auto TOLX = torch::tensor(std::numeric_limits<double>::epsilon(), torch::dtype(torch::kFloat64)).to(x.device());
        int M = x.r.size(0); // Batch size
        int N = x.r.size(1); // State space size
        auto xnorm = x.square().sum().sqrt();
        auto xcomp = TensorDual::ones_like(xnorm)*N;
        auto stpmax = STPMX * max(xnorm, xcomp);
        // Is the initial guess a root?
        auto test = f.abs().max(); // This is dimension M
        auto check = torch::zeros({M}, torch::dtype(torch::kBool)).to(x.device()); 
        auto p = TensorDual::zeros_like(f);
        auto g = TensorDual::zeros_like(x);
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
            //Recalculate the function and the jacobian
            auto f_copy = f.clone();

            
            f_copy.index_put_({m2}, func(x.index({m2}).contiguous(),
                                    params.index({m2}).contiguous()));
            f = f_copy;
            auto jac_copy = jac.clone();

            jac_copy.index_put_({m2}, jacfunc(x.index({m2}).contiguous(),
                                 params.index({m2}).contiguous()));
            jac = jac_copy;
            //Calculate the gradient
            auto g_copy = g.clone();
            g_copy.index_put_({m2}, TensorMatDual::einsum("mij, mi->mj", jac.index({m2}).contiguous(),
                                                             f.index({m2}).contiguous()));
            g = g_copy;
            auto xold_copy = xold.clone();
            xold_copy.index_put_({m2}, x.index({m2}).contiguous());
            xold = xold_copy;
            auto fold_copy = fold.clone();
            fold_copy.index_put_({m2}, f.index({m2}).contiguous());
            fold = fold_copy;
            auto Jold_copy = Jold.clone();
            Jold_copy.index_put_({m2}, J.index({m2}).contiguous());
            Jold = Jold_copy;
            // Since the jacobian has been recalcualted, we need to recompute the QR decomposition
            //auto [qt, r] = qrte(jac.index({check}).contiguous()); // This needs to be replaced with GMRES for vey high dimensional systems
            auto jacm2 = jac.index({m2}).contiguous();
            auto [qt, r] = qrted(jacm2);
            //auto [LU, P] = LUTeD(jacm2);
            auto pm2 = -f.index({m2}).contiguous();
            auto sollu = qrtedsolvev(qt, r, pm2);
            //auto sol = qrtesolvev(qt, r, pm2);
            //Test the solution
            //auto Jacsollu = torch::einsum("mij, mj->mi", {jacm2, sollu});
            //std::cerr << "Error from LU decomposition=" << (Jacsollu - pm2).norm(2, 1) << std::endl;
            //auto Jacsolqr = torch::einsum("mij, mj->mi", {jacm2, sol});
            //std::cerr << "Error from QR decomposition=" << (Jacsolqr - pm2).norm(2, 1) << std::endl;
            auto p_copy = p.clone();
            p_copy.index_put_({m2}, sollu);
            p = p_copy;
            // Update x, p and f
            auto xoldin = xold.index({m2}).contiguous();
            auto foldin = fold.index({m2}).contiguous();
            auto joldin = Jold.index({m2}).contiguous();
            auto gin = g.index({m2}).contiguous();
            auto pin = p.index({m2}).contiguous();
            auto stpmaxin = stpmax.index({m2}).contiguous();
            auto paramsin = params.index({m2}).contiguous();
            auto [xs, js, ps, checkupd] = lnsrchTeD(xoldin,
                                                   foldin,
                                                   joldin,
                                                   gin,
                                                   pin,
                                                   stpmaxin,
                                                   paramsin,
                                                   func);
            auto x_copy = x.clone();
            x_copy.index_put_({m2}, xs);
            x = x_copy;
            auto p_copy1 = p.clone();
            p_copy1.index_put_({m2}, ps);
            p = p_copy1;
            auto J_copy = J.clone();
            J_copy.index_put_({m2}, js);
            J = J_copy;
            auto check_copy = check.clone();
            check_copy.index_put_({m2}, checkupd);
            check = check_copy;
            auto test_copy = test.clone();
            test_copy.index_put_({m2}, f.index({m2}).abs().max());
            test = test_copy;
            auto m2_1 = m2 & (test < TOLF);
            if (m2_1.any().eq(true_t).item<bool>())
            {
              auto check_copy1 = check.clone();
              check_copy1.index_put_({m2_1}, false);
              check = check_copy1;
              auto m2_copy = m2.clone();
              m2_copy.index_put_({m2_1}, false);
              m2 = m2_copy;
            }
            

            // Need to recheck the flags since they have been updated
            auto m2_2 = m2 & check;
            if ((m2_2).any().equal(true_t))
            {
                //Checks for convergence on the gradient
                auto test_copy = test.clone();
                test_copy.index_put_({m2_2}, 0.0);
                test = test_copy;
                auto Jl = J.index({m2_2}).contiguous();
                auto den = max(Jl, 0.5 * TensorDual::ones_like(Jl) * N);
                auto xt = x.index({m2_2}).abs();
                auto ONE = TensorDual::ones_like(xt);
                auto rhs = TensorDual::einsum("mi,m->mi", {max(xt, ONE),  den.reciprocal()});
                auto temp = TensorDual::einsum("mi,mi->m",{g.index({m2_2}).abs(),rhs});
                auto test_copy1 = test.clone();
                test_copy1.index_put_({m2_2}, temp.max());
                test = test_copy1;
                auto check_copy1 = check.clone();
                check_copy1.index_put_({m2_2}, test.index({m2_2}) < TOLMIN);
                check = check_copy1;
                auto m2_copy = m2.clone();
                m2_copy.index_put_( {m2_2}, false); //We are done with this samples
                m2 = m2_copy; 

            }
            //Recheck for the check flags since they have been updated
            if (m2.eq(true_t).any().item<bool>())
            {
                // Check for small relative changes in x
                auto dx = (x.index({m2}) - xold.index({m2})).abs();
                auto xabs = x.index({m2}).abs();
                test.index_put_({m2}, (dx/xabs).max());
                if ( (test.index({m2}) < TOLX).any().item<bool>() )
                {
                    std::cerr << "Small relative changes in x detected" << std::endl;
                    //No need to continue for these samples
                    m2.index_put_({m2.clone()}, false);    
                }
                auto m2_copy = m2.clone();
                if (m2_copy.index({m2}).eq(true_t).any().item<bool>())
                {
                    m2_copy.index_put_({m2.clone()}, ~(test.index({m2}) < TOLX));
                }
                
                m2 = m2_copy;
            }
        }
        
        return std::make_tuple(x, check);
    }
    } // namespace nlp
    
} // namespace janus

#endif // newt_HPP_INCLUDED