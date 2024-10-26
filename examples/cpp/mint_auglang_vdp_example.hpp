#ifndef MINT_AUGLANG_VDP_EXAMPLE_HPP
#define MINT_AUGLANG_VDP_EXAMPLE_HPP
/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#define HEADER_ONLY
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/autograd.h>
#include <janus/radauted.hpp>

using namespace janus;
namespace py = pybind11;

namespace janus
{
  namespace nlp
  {
    namespace examples
    {
      namespace vdp
      {
        namespace mint // min time optimal control
        {
          namespace auglang // Augmented Langrangian
          {

            /**
             * Radau example using the Van der Pol oscillator
             * Using the Hamiltonian with dual number approach to calcuate the dynamics and
             * the Jacobian
             */
            using Slice = torch::indexing::Slice;
            // Global parameters with default values
            torch::Tensor x1f = torch::tensor({-1.0}, {torch::kFloat64});
            torch::Tensor x2f = torch::tensor({-1.0}, {torch::kFloat64});
            torch::Tensor x10 = torch::tensor({1.0}, {torch::kFloat64});
            torch::Tensor x20 = torch::tensor({10.0}, {torch::kFloat64});
            int MaxNbrStep = 10000; // Limit the number of steps to avoid long running times
            double u1min = 0.0, u2min = 1.0, u3min = 0.0;
            double u1max = 0.0, u2max = 100.0, u3max = 0.0;


            TensorDual Jfunc(const TensorDual &F)
            {
              return F.pow(2).sum() / 2.0;
            }

            torch::Tensor Jfunc(const torch::Tensor &F)
            {
              return F.pow(2).sum() / 2.0;
            }

            void set_xf(const torch::Tensor &x1, torch::Tensor &x2)
            {
              x1f = x1;
              x2f = x2;
            }

            void set_x0(const torch::Tensor &x1, torch::Tensor &x2)
            {
              x10 = x1;
              x20 = x2;
            }


          void set_ulimits(const double &u1minval, const double &u2minval, const double &u3minval,
                           const double &u1maxval, const double &u2maxval, const double &u3maxval)
          {
            u1min = u1minval;
            u2min = u2minval;
            u3min = u3minval;
            u1max = u1maxval;
            u2max = u2maxval;
            u3max = u3maxval;
          }

          std::tuple<TensorDual, TensorDual, TensorDual> calc_control(const TensorDual &p1,
                                  const TensorDual &p2,
                                  const TensorDual &x1,
                                  const TensorDual &x2)
          {
            auto m = p1 < 0.0;
            auto u1star = TensorDual::zeros_like(p1);
            u1star.index_put_({m}, u1max);
            u1star.index_put_({~m}, u1min);
            auto m2 = p2*(1-x1*x1)*x2 < 0.0;
            auto u2star = TensorDual::zeros_like(p2);
            u2star.index_put_({m2}, u2max);
            u2star.index_put_({~m2}, u2min);
            auto m3 = p2 < 0.0;
            auto u3star = TensorDual::zeros_like(p2);
            u3star.index_put_({m3}, u3max);
            u3star.index_put_({~m3}, u3min);
            return {u1star, u2star, u3star};
          }

          std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> calc_control(const torch::Tensor &p1,
                                     const torch::Tensor &p2,
                                     const torch::Tensor &x1,
                                     const torch::Tensor &x2)
          {
            auto u1star = torch::zeros_like(p1);
            auto m1 = p1 < 0.0;
            u1star.index_put_({m1}, u1max);
            u1star.index_put_({~m1}, u1min);
            auto u2star = torch::zeros_like(p2);
            auto m2 = p2*(1-x1*x1)*x2 < 0.0;
            u2star.index_put_({m2}, u2max);
            u2star.index_put_({~m2}, u2min);
            auto u3star = torch::zeros_like(p2);
            auto m3 = p2 < 0.0;
            u3star.index_put_({m3}, u3max);
            u3star.index_put_({~m3}, u3min);
            return {u1star, u2star, u3star};
          }


          


          std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> calc_ustar(const torch::Tensor &p1,
                                     const torch::Tensor &p2,
                                     const torch::Tensor &x1,
                                     const torch::Tensor &x2)
          {
            return calc_control(p1, p2, x1, x2);
          }


            /**
             * Dynamics calculated for the augmented Langrangian formulation
             */
            TensorDual vdpdyns_Lang(const TensorDual &t,
                                    const TensorDual &y,
                                    const TensorDual &params)
            {
              // auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
              auto x = y.index({Slice(), Slice(2, 4)});
              auto p = y.index({Slice(), Slice(0, 2)});
              // std::cerr << "Control = " << ustar << std::endl;
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto p2 = p.index({Slice(), Slice(1, 2)});
              auto p1 = p.index({Slice(), Slice(0, 1)});
              auto [u1star, u2star, u3star] = calc_control(p1.r,p2.r,x1.r,x2.r);
              std::cerr << "u1star = " << u1star << std::endl;
              std::cerr << "u2star = " << u2star << std::endl;
              std::cerr << "u3star = " << u3star << std::endl;

              //H=p1*x2+p2*mu*(1-x1*x1)*x2-p2*x1+p2*ustar+1;
              auto dp1dt = p2 * u2star * (-2 * x1) * x2 - p2;
              auto dp2dt = p1 + p2 * u2star * (1 - x1 * x1);
              auto dx1dt = x2+u1star;
              auto dx2dt = u2star * (1 - x1 * x1) * x2 - x1 + u3star;

              auto real_dyns = TensorDual::cat({dp1dt, dp2dt, dx1dt, dx2dt});

              return real_dyns;
            }


            TensorMatDual jac_Lang(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              // auto jacv = evalJacDual<double>(y, W, hamiltonian);
              // return jacv;
              auto x = y.index({Slice(), Slice(2, 4)});
              auto p = y.index({Slice(), Slice(0, 2)});
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto p1 = p.index({Slice(), Slice(0, 1)});
              auto p2 = p.index({Slice(), Slice(1, 2)});
      
              auto [u1star, u2star, u3star] = calc_control(p1.r, p2.r, x1.r, x2.r);
  
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 4, 4}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 4, 4, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);
              //p2 * u2star * (-2 * x1) * x2 - p2;
              jac.index_put_({Slice(), Slice(0, 1), 1}, u2star * (-2 * x1) * x2 - one);
              jac.index_put_({Slice(), Slice(0, 1), 2}, -p2 * u2star * 2 * x2);
              jac.index_put_({Slice(), Slice(0, 1), 3}, -p2 * u2star * 2 * x1);
              // p1 + p2 * u2star * (1 - x1 * x1);
              jac.index_put_({Slice(), Slice(1, 2), 0}, one);
              jac.index_put_({Slice(), Slice(1, 2), 1}, u2star * (1 - x1 * x1));
              jac.index_put_({Slice(), Slice(1, 2), 2}, p2 * u2star * (-2 * x1));
              // x2+u1star;
              jac.index_put_({Slice(), Slice(2, 3), 3}, one);
              // u2star*((1-x1*x1)*x2)-x1+u3star
              jac.index_put_({Slice(), Slice(3, 4), 2}, u2star * (-2 * x1 * x2) - one);
              jac.index_put_({Slice(), Slice(3, 4), 3}, u2star * ((1 - x1 * x1)));
              
              return jac;
            }





            /**
             * Radau example using the Van der Pol oscillator
             * with sensitivity calculations utilizing dual numbers
             * to calculate the gradients of the augmented Langrangian function
             * The function returns the residuals of the expected
             * end state wrt x1f x2f and final Hamiltonian value
             * using p20 and tf as the input variables (x)
             * The relationship is defined by the necessary conditions
             * of optimality as defined by the Variational approach to
             * optimal control
             */
            std::tuple<torch::Tensor, torch::Tensor, 
                       torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor> 
                       mint_auglangr_propagate(const torch::Tensor &xic,
                                               const torch::Tensor &x,
                                               const torch::Tensor &lambdap,
                                               const torch::Tensor &mup,           
                                               const torch::Tensor &params,
                                               const bool rescale)
            {
              std::cerr << "Starting the augmented Langrangian calculation" << std::endl;
              std::cerr << "xic = " << xic << std::endl;
              std::cerr << "x = " << x << std::endl;
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 4; // Length of the state space vector in order [p1, p2, x1, x2]
              int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
              double rtol = 1e-3;
              double atol = 1e-6;
              if (params.dim() == 1)
              {
                rtol = params.index({0}).item<double>();
                atol = params.index({1}).item<double>();
              }
              else
              {
                rtol = params.index({Slice(), 0}).item<double>();
                atol = params.index({Slice(), 1}).item<double>();
              }
              auto p10td = TensorDual(x.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              p10td.d.index_put_({Slice(), 0, 0}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto p20td = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              p20td.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto x10td = TensorDual(xic.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              x10td.d.index_put_({Slice(), 0, 2}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto x20td = TensorDual(xic.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              x20td.d.index_put_({Slice(), 0, 3}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto ft = TensorDual(x.index({Slice(), Slice(2, 3)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 4}, 1.0); // Set the dependency to itself

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

              y.index_put_({Slice(), Slice(0, 1)}, p10td);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, p20td);       // p2
              y.index_put_({Slice(), Slice(2, 3)}, one*x10td);   // x1
              y.index_put_({Slice(), Slice(3, 4)}, one*x20td);   // x2
              auto y0 = y.clone();                             // Copy of the initial conditions

              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              TensorDual tspan = TensorDual(torch::rand({M, 2}, x.options()), torch::zeros({M, 2, D}, x.options()));
              tspan.r.index_put_({Slice(), Slice(0, 1)}, 0.0);
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.index_put_({Slice(), Slice(1, 2)}, ft);
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
              options.RelTol = torch::tensor({rtol}, x.options());
              options.AbsTol = torch::tensor({atol}, x.options());

              options.MaxNbrStep = MaxNbrStep;
              auto tspanc = tspan.clone();
              auto yc = y.clone();
              auto paramsc = TensorDual::empty_like(y);

              janus::RadauTeD r(vdpdyns_Lang, jac_Lang, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
              // Call the solve method of the Radau5 class
              auto rescode = r.solve();
              //Check the return codes
              auto m = rescode != 0;
              if ( m.any().item<bool>())
              {
                //The solver fails gracefully so we just provide a warning message here
                std::cerr << "Solver failed to converge for some samples" << std::endl;
              }
              //Record the projected final values
              auto p1pf = r.y.index({Slice(), Slice(0, 1)});
              auto p2pf = r.y.index({Slice(), Slice(1, 2)});
              auto x1pf = r.y.index({Slice(), Slice(2, 3)});
              auto x2pf = r.y.index({Slice(), Slice(3, 4)});
              auto yend = r.y.index({Slice(), Slice(0, 4)});
              //auto ustarpf = calc_control(p1pf, p2pf, x1pf, x2pf);
              auto p0 = y0.r.index({Slice(), Slice(0, 2)});
              //Rescale the constraints to avoid huge fluctuations in the gradients
              auto c1x = x1pf + x1pf.abs();
              if (rescale)
              {
                auto normc1xd = torch::norm(c1x.d.index({Slice(), 0, Slice(0,2)}), 2, {1});
                auto scale1 = (1+(1+normc1xd).log()).reciprocal();
                c1x = TensorDual::einsum("mi, m->mi", c1x, scale1);
              }
              auto c2x = x2pf;
              if ( rescale)
              {
                auto normc2xd = torch::norm(c2x.d.index({Slice(), 0, Slice(0,2)}), 2, {1});
                auto scale2 = (1+(1+normc2xd).log()).reciprocal();
                c2x = TensorDual::einsum("mi, m->mi",c2x,scale2);
              }
              //auto [u1starf, u2starf] = calc_control(p1pf, p2pf, x1pf, x2pf);

              

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(0,1)}),c1x)
                       - TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(1,2)}),c2x)
                       +(one+ 0.5*TensorDual::einsum("mi, mi->mi",mup,c1x.square())
                       + 0.5*TensorDual::einsum("mi, mi->mi",mup,c2x.square())).log();


              auto grads = torch::zeros_like(x);
              auto dJdp1 =f.d.index({Slice(), 0, Slice(0, 1)});
              auto dJdp1scale = (1.0+dJdp1.norm()).log();
              dJdp1 = dJdp1/dJdp1scale;
               auto dJdp2 =f.d.index({Slice(), 0, Slice(1, 2)});
              auto dJdp2scale = (1.0+dJdp2.norm()).log();
              dJdp2 = dJdp2/dJdp2scale;
              auto dJdft =f.d.index({Slice(), 0, Slice(2, 3)});
              auto dJdftscale = (1.0+dJdft.norm()).log();
              dJdft = dJdft/dJdftscale;

              grads.index_put_({Slice(), 0}, dJdp1); // p1
              grads.index_put_({Slice(), 1}, dJdp2); // p2
              grads.index_put_({Slice(), 2}, dJdft); // ft
              
              
              auto errors = torch::cat({c1x.r, c2x.r}, 1);
              auto error_norm = errors.norm(2, {1}, true);
              //The jacobian is block diagonal
              auto jac = torch::zeros({M, 2, 3}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, c1x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, c1x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(0, 1), Slice(2, 3)}, c1x.d.index({Slice(), Slice(0,1), Slice(4, 5)})); // ft

              jac.index_put_({Slice(), Slice(1, 2), Slice(0, 1)}, c2x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(1, 2), Slice(1, 2)}, c2x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(1, 2), Slice(2, 3)}, c2x.d.index({Slice(), Slice(0,1), Slice(4, 5)})); // ft
           
              return std::make_tuple(f.r, grads, yend.r, errors, error_norm, jac);
            }

            /**
             * Radau ODE example using the Van der Pol oscillator
             * with sensitivity calculations utilizing dual numbers
             * to calculate the gradients of the constraints
             * The function returns the residuals of the expected
             * end state wrt x1f x2f and final Hamiltonian value
             * using p20 and tf as the input variables (x)
             * The relationship is defined by the necessary conditions
             * of optimality as defined by the Variational approach to
             * optimal control
             * Function returns the objective and the constraint violations and the
             * jacobian of the constraints
             */
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mint_propagate(const torch::Tensor &x,
                                                                             const torch::Tensor &params)
            {
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 4; // Length of the state space vector in order [p1, p2, x1, x2, LI]
              int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, L, tf]
              double rtol = 1e-6;
              double atol = 1e-9;
              if (params.dim() == 1)
              {
                rtol = params.index({0}).item<double>();
                atol = params.index({1}).item<double>();
              }
              else
              {
                rtol = params.index({Slice(), 0}).item<double>();
                atol = params.index({Slice(), 1}).item<double>();
              }
              auto p10td = TensorDual(x.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              p10td.d.index_put_({Slice(), 0, 0}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto p20td = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              p20td.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto x10td = TensorDual(x10.clone(), torch::zeros({M, 1, D}, x.options()));
              x10td.d.index_put_({Slice(), 0, 2}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto x20td = TensorDual(x20.clone(), torch::zeros({M, 1, D}, x.options()));
              x20td.d.index_put_({Slice(), 0, 3}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto ustartd = calc_control(p10td, p20td, x10td, x20td);
              //p10td = calc_p1(p10td, p20td, x10td, x20td, ustartd);
              //H=p1*x2+p2*mu*(1-x1*x1)*x2-p2*x1+p2*u+0.5*W*u*u+1;
              auto ft = TensorDual(x.index({Slice(), Slice(2, 3)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 4}, 1.0); // Set the dependency to itself
              std::cerr << "ft=";
              janus::print_dual(ft);

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));


              y.index_put_({Slice(), Slice(0, 1)}, p10td);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, p20td);       // p2
              y.index_put_({Slice(), Slice(2, 3)}, one*x10td); // x1
              y.index_put_({Slice(), Slice(3, 4)}, one*x20td); // x2
              auto y0 = y.clone();                             // Copy of the initial conditions

              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              TensorDual tspan = TensorDual(torch::rand({M, 2}, x.options()), torch::zeros({M, 2, D}, x.options()));
              tspan.r.index_put_({Slice(), Slice(0, 1)}, 0.0);
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.index_put_({Slice(), Slice(1, 2)}, ft);
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
              options.RelTol = torch::tensor({rtol}, x.options());
              options.AbsTol = torch::tensor({atol}, x.options());

              options.MaxNbrStep = MaxNbrStep;
              auto tspanc = tspan.clone();
              auto yc = y.clone();
              auto paramsc = TensorDual::empty_like(y);
              std::cerr << "Setting tolerances rtol=" << rtol << " atol=" << atol << std::endl;

              janus::RadauTeD r(vdpdyns_Lang, jac_Lang, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
              // Call the solve method of the Radau5 class
              auto rescode = r.solve();
              //Check the return codes
              auto m = rescode != 0;
              if ( m.any().item<bool>())
              {
                //The solver fails gracefully so we just provide a warning message here
                std::cerr << "Solver failed to converge for some samples" << std::endl;
              }
              auto p1pf = r.y.index({Slice(), Slice(0, 1)});
              auto p2pf = r.y.index({Slice(), Slice(1, 2)});
              auto x1pf = r.y.index({Slice(), Slice(2, 3)});
              auto x2pf = r.y.index({Slice(), Slice(3, 4)});
              auto ustartf = (-p1pf * x2pf - p2pf * mu * (1 - x1pf * x1pf) * x2pf + 
                        p2pf * x1pf-1)/p2pf; 

              // Now calculate the boundary conditionsx
              auto x1delta = (r.y.index({Slice(), Slice(2, 3)}) - x1f);
              auto x2delta = (r.y.index({Slice(), Slice(3, 4)}) - x2f);
              auto p1delta = (r.y.index({Slice(), Slice(0, 1)}));
              auto p2delta = (r.y.index({Slice(), Slice(1, 2)}));
              //H = p1*x2 + p2 * mu * (1 - x1 * x1) * x2 - p2 * x1 + p2*ustar + 1;
              auto Hf = p1pf * x2pf + p2pf * mu * (1 - x1pf * x1pf) * x2pf - 
                        p2pf * x1pf + p2pf * ustartf +1;

              //std::cerr << "x2delta=";
              //janus::print_dual(x2delta);
              auto deltas = torch::cat({p1delta.r, p2delta.r, x1delta.r, x2delta.r}, 1);  //Boundary conditions

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = ft+r.y.index({Slice(), Slice(4, 5)});
              std::cerr << "For input x = " << x << std::endl;
              std::cerr << "f = " << f << std::endl;
              janus::print_dual(f);
              auto grads = torch::zeros_like(x);
              auto dJdp1 =f.d.index({Slice(), 0, Slice(0, 1)});
              auto dJdp1scale = (1.0+dJdp1.norm()).log();
              dJdp1 = dJdp1/dJdp1scale;
              auto dJdp2 =f.d.index({Slice(), 0, Slice(1, 2)});
              auto dJdp2scale = (1.0+dJdp2.norm()).log();
              dJdp2 = dJdp2/dJdp2scale;
              auto dJdft =f.d.index({Slice(), 0, Slice(2, 3)});
              auto dJdftscale = (1.0+dJdft.norm()).log();
              dJdft = dJdft/dJdftscale;
              grads.index_put_({Slice(), Slice(0, 1)}, dJdp1); // p1
              grads.index_put_({Slice(), Slice(1, 2)}, dJdp2); // p2
              grads.index_put_({Slice(), Slice(2, 3)}, dJdft); // ft

              //Rescale the gradients
              // The hamiltonian is zero at the terminal time
              // because this is a minimum time problem
              auto jac = torch::zeros({M, 4, 3}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, p1delta.d.index({Slice(), 0, Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, p1delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(0, 1), Slice(2, 3)}, p1delta.d.index({Slice(), 0, Slice(4, 5)})); // ft

              jac.index_put_({Slice(), Slice(1, 2), Slice(0, 1)}, p2delta.d.index({Slice(), 0, Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(1, 2), Slice(1, 2)}, p2delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(1, 2), Slice(2, 3)}, p2delta.d.index({Slice(), 0, Slice(4, 5)})); // ft

              jac.index_put_({Slice(), Slice(2, 3), Slice(0, 1)}, x1delta.d.index({Slice(), 0, Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(2, 3), Slice(1, 2)}, x1delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(2, 3), Slice(2, 3)}, x1delta.d.index({Slice(), 0, Slice(4, 5)})); // ft


              jac.index_put_({Slice(), Slice(3, 4), Slice(0, 1)}, x2delta.d.index({Slice(), 0, Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(3, 4), Slice(1, 2)}, x2delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(3, 4), Slice(2, 3)}, x2delta.d.index({Slice(), 0, Slice(4, 5)})); // ft

              auto res = std::make_tuple(f.r, grads, deltas, jac);
              return res;
            }

          }
        }
      }
    }
  }

} // namespace janus

#endif