#ifndef LNSEARCH_TED_HPP_INCLUDED
#define LNSEARCH_TED_HPP_INCLUDED

#include <torch/torch.h>
#include <iostream>
#include <initializer_list>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>

namespace janus
{

    TensorDual Jfunc(const TensorDual &F)
    {

        return F.pow(2).sum()/2.0;
        
    }

    /**
     * Perform line search in parallel for a batch of points
     */
    std::tuple<TensorDual, TensorDual, TensorDual, torch::Tensor> lnsrchTeD(TensorDual &xold,
                                                                        TensorDual &fold,
                                                                        TensorDual &Jold,
                                                                        TensorDual &g,
                                                                        TensorDual &p,
                                                                        TensorDual &stpmax,
                                                                        TensorDual &params,
                                                                        TensorDual &xmin,
                                                                        TensorDual &xmax,
                                                                        std::function<TensorDual(TensorDual &, TensorDual &)> func)
    {
        const torch::Tensor ALF = torch::tensor({1.0e-4}, torch::dtype(torch::kFloat64)).to(xold.device());
        const torch::Tensor TOLX = torch::tensor({std::numeric_limits<double>::epsilon()}).to(xold.device());
        const TensorDual ZERO = TensorDual::zeros_like(xold);
        TensorDual x = xold.clone();
        int M = xold.r.size(0); // Number of samples
        int N = xold.r.size(1); // Number of dimensions
        int D = xold.d.size(2); // Number of dual numbers
        assert(xold.r.size(0) == fold.r.size(0) && xold.r.size(0) == g.r.size(0) 
                                            && xold.r.size(0) == p.r.size(0) 
                                            && "All inputs must have the same batch size");
        auto true_t = torch::tensor({true}, torch::dtype(torch::kBool)).to(xold.device());
        auto false_t = torch::tensor({false}, torch::dtype(torch::kBool)).to(xold.device());
        auto sum = p.square().sum().sqrt(); // Sum across the batch dimension while retaining the batch dimension
        auto m1 = sum > stpmax;              // This will retain the batch dimension
        if (m1.eq(true_t).any().item<bool>())
        {
            auto fac = stpmax.index({m1}).contiguous() / sum.index({m1}).contiguous();
            auto pm1 = p.index({m1}).contiguous();
            p.index_put_({m1}, TensorDual::einsum("mi,mk->mi", pm1, fac));
        }
        auto slope = TensorDual::einsum("mi,mi->mi", {g,  p});
        if ((slope >= 0).any().equal(true_t))
        {
            std::cerr << "Positive slope in lnsrch for samples" << (slope >= 0).nonzero() << std::endl;
        }
        auto xolda = xold.abs();
        auto ONES = TensorDual::ones_like(xolda);
        auto test = (p.abs() / max(xolda, ONES));
        auto alamin = (TOLX / test);
        auto cond = torch::zeros({M}, torch::dtype(torch::kFloat64)).to(xold.device());
        auto tmplam = TensorDual(torch::zeros({M, N}, torch::dtype(torch::kFloat64)).to(xold.r.device()),
                                torch::zeros({M, N, D}, torch::dtype(torch::kFloat64)).to(xold.r.device()));
        auto J2 = TensorDual(torch::zeros({M,1}, torch::dtype(torch::kFloat64)).to(xold.r.device()),
                             torch::zeros({M,1,D}, torch::dtype(torch::kFloat64)).to(xold.r.device()));
        auto alam = TensorDual(torch::ones({M, N}, torch::dtype(torch::kFloat64)).to(xold.device()),
                                torch::zeros({M, N, D}, torch::dtype(torch::kFloat64)).to(xold.device()));   
        auto alam2 = TensorDual::zeros_like(alam);
        auto a = TensorDual(torch::zeros({M, N}, torch::dtype(torch::kFloat64)).to(xold.device()),
                            torch::zeros({M, N, D}, torch::dtype(torch::kFloat64)).to(xold.device()));  
        auto b = TensorDual(torch::zeros({M, N}, torch::dtype(torch::kFloat64)).to(xold.device()),
                            torch::zeros({M, N, D}, torch::dtype(torch::kFloat64)).to(xold.device()));  
        auto disc = TensorDual(torch::zeros({M,N}, torch::dtype(torch::kFloat64)).to(xold.device()),
                                 torch::zeros({M,N,D}, torch::dtype(torch::kFloat64)).to(xold.device()));
        auto check = torch::zeros({M}, torch::dtype(torch::kBool)).to(xold.device());

        auto Jres = TensorDual::zeros_like(Jold);
        // This is an arbitrary mask to allow for checks while the loop is running
        // If return or break conditions are encountered the loop will terminate
        torch::Tensor m2 = torch::ones({M}, torch::dtype(torch::kBool)).to(xold.device());
        while (m2.eq(true_t).any().item<bool>())
        {
            auto xupdt = xold.index({m2}).contiguous() +
                         TensorDual::einsum("mi,mi->mi", {alam.index({m2}).contiguous(), 
                                                         p.index({m2}).contiguous()});
            // Check if the new x is within bounds
            auto m2_1_1 = (xupdt < xmin.index({m2})).all(1);    
            if (m2_1_1.eq(true_t).any().item<bool>())
            {
                xupdt.index_put_({m2_1_1}, xmin.index({m2_1_1}));
                //also update the lambda factor
                alam.index_put_({m2_1_1}, (xmin.index({m2_1_1}) - xold.index({m2_1_1})) / p.index({m2_1_1}));
            }
            auto m2_1_2 = (xupdt > xmax.index({m2}));
            if (m2_1_2.eq(true_t).any().item<bool>())
            {
                xupdt.index_put_({m2_1_2}, xmax.index({m2_1_2}));
                //also update the lambda factor
                alam.index_put_({m2_1_2}, (xmax.index({m2_1_2}) - xold.index({m2_1_2})) / p.index({m2_1_2}));
            }
            x.index_put_({m2}, xupdt);
            auto xm2 = x.index({m2}).contiguous();
            auto paramsm2 = params.index({m2}).contiguous();
            Jres.index_put_({m2}, Jfunc(func(xm2, paramsm2)));
            // This is possibly a multi-dimensional function so we convert to scalar

            auto m2_1 = m2 & (alam < alamin).all(1);
            if (m2_1.eq(true_t).any().item<bool>())
            {
                x.index_put_({m2_1}, xold.index({m2_1}));
                check.index_put_({m2_1}, true);
                m2.index_put_({m2_1}, false); // We are done for these samples
            }
            // Recheck m2 before the sufficiency condition
            if (m2.eq(true_t).any().item<bool>())
            {

                auto suff = (Jres.index({m2}).contiguous() <= (Jold.index({m2}).contiguous() +
                                                               ALF * alam.index({m2}).contiguous()*
                                                               slope.index({m2}).contiguous()).sum());
                m2.index_put_({m2.clone()}, ~suff); // Sufficiency condition
            }
            // Check m2 after the sufficiency condition
            auto m2_3 = m2 & (alam == 1.0).all(1);
            if (m2_3.any().eq(true_t).any().item<bool>())
            {
                auto den = (2.0 * (Jres.index({m2_3}).contiguous() -
                                   Jold.index({m2_3}).contiguous() -
                                   slope.index({m2_3}).contiguous()));
                tmplam.index_put_({m2_3}, -slope.index({m2_3}).contiguous() / den);
            }
            auto m2_4 = m2 & (alam != 1.0).all(1);
            if (m2_4.any().eq(true_t).any().item<bool>())
            {
                std::cerr << "m2_4 " << m2_4 << std::endl;
                std::cerr << "Jres=";
                janus::print_dual(Jres);
                std::cerr << "Jold=";
                janus::print_dual(Jold);
                auto rhs1 = Jres.index({m2_4}).contiguous() -
                            Jold.index({m2_4}).contiguous() -
                            (alam.index({m2_4}).contiguous()* 
                            slope.index({m2_4}).contiguous()).sum();
                auto rhs2 = J2.index({m2_4}).contiguous() -
                            Jold.index({m2_4}).contiguous() -
                            (alam2.index({m2_4}).contiguous()* 
                            slope.index({m2_4}).contiguous()).sum();
                a.index_put_({m2_4}, (rhs1 / alam.index({m2_4}).contiguous().square() -
                                      rhs2 / alam2.index({m2_4}).contiguous().square()) /
                                      (alam.index({m2_4}).contiguous()) -
                                       alam2.index({m2_4}).contiguous());
                b.index_put_({m2_4}, (-alam2.index({m2_4}).contiguous() * rhs1 / 
                                       (alam.index({m2_4}).contiguous().square())  +
                                        alam.index({m2_4}).contiguous() * rhs2 / 
                                        (alam2.index({m2_4}).contiguous().square())) /
                                        (alam.index({m2_4}).contiguous() - alam2.index({m2_4}).contiguous()));


                auto m2_4_1 = m2_4 & (a == 0).all(1);
                if (m2_4_1.eq(true_t).any().item<bool>())
                {
                    tmplam.index_put_({m2_4_1}, -slope.index({m2_4_1}).contiguous() / 
                                                (2.0 * b.index({m2_4_1}).contiguous()));
                }
                auto m2_4_2 = m2_4 & (a != 0.0).all(1);
                if (m2_4_2.eq(true_t).any().item<bool>())
                {
                        // disc=b*b-3.0*a*slope;
                    disc.index_put_({m2_4_2}, b.index({m2_4_2}).contiguous().square() -
                                             3.0 * a.index({m2_4_2}).contiguous() *
                                             slope.index({m2_4_2}).contiguous());
                    std::cerr << "disc.index({m2_4_2}) < 0=" << (disc < 0).all(1) << std::endl;
                    //Cap the discriminator so it is never above a large number
                    if ((disc > 1.0e+10).any().item<bool>())
                    {
                        disc.index_put_({disc > 1.0e+10}, 1.0e+10);
                    }
                    if ((disc < -1.0e+10).any().item<bool>())
                    {
                        disc.index_put_({disc < -1.0e+10}, -1.0e+10);
                    }
                    std::cerr << "disc " << disc << std::endl;


                    auto m2_4_2_1 = m2_4_2 & (disc < 0).all(1);
                    if (m2_4_2_1.eq(true_t).any().item<bool>())
                    {
                      tmplam.index_put_({m2_4_2_1}, 0.5 * alam.index({m2_4_2_1}).contiguous());
                    }
                    auto m2_4_2_2 = m2_4_2 & 
                                   (disc >= 0.0).all(1) & 
                                   (b <= 0).all(1);
                    if (m2_4_2_2.eq(true_t).any().item<bool>())
                    {
                      // tmplam=(-b+sqrt(disc))/(3.0*a);
                      tmplam.index_put_({m2_4_2_2}, (-b.index({m2_4_2_2}).contiguous() +
                                                     disc.index({m2_4_2_2}).contiguous().sqrt()) /
                                                     (3.0 * a.index({m2_4_2_2}).contiguous()));
                    }
                    auto m2_4_2_3 = m2_4_2 & 
                                    (b > 0.0).all(1) & 
                                    (disc >= 0).all(1);
                    if (m2_4_2_3.eq(true_t).any().item<bool>())
                    {
                        // tmplam=-slope/(b+sqrt(disc));
                        tmplam.index_put_({m2_4_2_3}, -slope.index({m2_4_2_3}).contiguous() /
                                                     (b.index({m2_4_2_3}).contiguous() +
                                                      disc.index({m2_4_2_3}).contiguous().sqrt()));
                    }
                    auto m2_4_2_4 = m2_4_2 & 
                                    (tmplam > 0.5 * alam).all(1);
                    if (m2_4_2_4.eq(true_t).any().item<bool>())
                    {
                      tmplam.index_put_({m2_4_2_4}, 0.5 * alam.index({m2_4_2_4}).contiguous());
                    }

                }
            }
            
            if (m2.eq(true_t).any().item<bool>())
            {
                alam2.index_put_({m2}, alam.index({m2}));
                J2.index_put_({m2}, Jres.index({m2}));
                auto tmplamm2 = tmplam.index({m2});
                auto alamm2 = alam.index({m2}) * 0.1;
                alam.index_put_({m2}, max(tmplamm2, alamm2));
            }
        }
        return std::make_tuple(x, Jres, p, check);
    }

} // namespace janus
#endif // LNSEARCH_HPP_INCLUDED