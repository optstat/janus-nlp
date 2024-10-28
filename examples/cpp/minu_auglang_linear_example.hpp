#ifndef MINT_AUGLANG_LINEAR_EXAMPLE_HPP
#define MINT_AUGLANG_LINEAR_EXAMPLE_HPP
/**
 * Use the Linear 1D example from Ross 2.9.1
 * The system minimizes 1/2 \int u^2 dt
 * subject to the dynamics
 * dx/dt = a*x + b*u
 * 
 * where the control effort is minimized
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
      namespace linear
      {
        namespace minu // min time optimal control
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
            torch::Tensor x1f = torch::tensor({1.0}, {torch::kFloat64});
            torch::Tensor x10 = torch::tensor({1.0}, {torch::kFloat64});
            double a = 1.0;
            double b = 1.0;
            double ft = 1.0;
            int MaxNbrStep = 10000; // Limit the number of steps to avoid long running times
 

            void set_xf(const torch::Tensor &x1)
            {
              x1f = x1;
            }

            void set_x0(const torch::Tensor &x1)
            {
              x10 = x1;
            }

            void set_a(double a_val)
            {
              a = a_val;
            }

            void set_b(double b_val)
            {
              b = b_val;
            }

            void set_ft(double ft_val)
            {
              ft = ft_val;
            }

          /**
           * The Hamiltonian in this case is
           * H = p1*a*x1+p1*b*u+0.5*u^2
           * ustar = -p1*b
           */
          TensorDual calc_control(const TensorDual &p1,
                                  const TensorDual &x1)
          {
            auto ustar = -p1*b;
            return ustar;
          }


          torch::Tensor calc_control(const torch::Tensor &p1,
                                     const torch::Tensor &x1)
          {
            auto ustar = -p1*b;
            return ustar;
          }



            /**
             * Dynamics calculated for the augmented Langrangian formulation
             */
            TensorDual lineardyns_Lang(const TensorDual &t,
                                       const TensorDual &y,
                                       const TensorDual &params)
            {
              // auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
              auto x1 = y.index({Slice(), Slice(1, 2)});
              auto p1 = y.index({Slice(), Slice(0, 1)});
              auto J  = y.index({Slice(), Slice(2, 3)});
              auto one = TensorDual::ones_like(x1);
              //The control is treated as an independent variable
              auto u1star = calc_control(p1.r, x1.r);
              auto dp1dt = a*p1;
              auto dx1dt = a*x1 +b*u1star;
              auto dJdt = 0.5*u1star.square()*one;

              auto real_dyns = TensorDual::cat({dp1dt, dx1dt, dJdt});

              return real_dyns;
            }


            TensorMatDual jac_Lang(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              // auto jacv = evalJacDual<double>(y, W, hamiltonian);
              // return jacv;
              auto x1 = y.index({Slice(), Slice(1, 2)});
              auto p1 = y.index({Slice(), Slice(0, 1)});
              auto J = y.index({Slice(), Slice(2, 3)});
      
              auto u1star = calc_control(p1.r, x1.r);
  
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 3, 3}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 3, 3, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);
              //-a*p1
              jac.index_put_({Slice(), Slice(0, 1), 0}, one*a);
              //a*x1 +b*u1star;
              jac.index_put_({Slice(), Slice(1, 2), 1}, one*a);
              
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
                       torch::Tensor> 
                       minu_auglangr_propagate(const torch::Tensor &xic,
                                               const torch::Tensor &x,  //This contains the costate and final time
                                               const torch::Tensor &lambdap,
                                               const torch::Tensor &mup,           
                                               const torch::Tensor &params)
            {
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 3; // Length of the state space vector in order [p1, x1, J]
              int D = 4; // Length of the dual vector in order [p1, x1, J, tf]
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
              auto x10td = TensorDual(xic.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              x10td.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto ft = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 3}, 1.0); // Set the dependency to itself

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

              y.index_put_({Slice(), Slice(0, 1)}, p10td);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, x10td);   // x1
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

              janus::RadauTeD r(lineardyns_Lang, jac_Lang, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
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
              auto x1pf = r.y.index({Slice(), Slice(1, 2)});

              auto c1x = x1pf - x1f;

              //auto [u1starf, u2starf] = calc_control(p1pf, p2pf, x1pf, x2pf);

              

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(0,1)}),c1x)
                       +(one+ 0.5*TensorDual::einsum("mi, mi->mi",mup,c1x.square())).log();


              auto grads = torch::zeros_like(x);
              grads.index_put_({Slice(), 0}, f.d.index({Slice(), 0, 0})); // p1
              grads.index_put_({Slice(), 2}, f.d.index({Slice(), 0, 1})); // ft
              
              
              auto errors = torch::cat({c1x.r}, 1);
              auto error_norm = torch::cat({c1x.r}, 1).norm();
              //The jacobian is block diagonal
              auto jac = torch::zeros({M, 1, 2}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, c1x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, c1x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // ft

           
              return std::make_tuple(f.r, grads, errors, error_norm, jac);
            }


          }
        }
      }
    }
  }

} // namespace janus

#endif