#ifndef MINU_AUGLANG_LINEAR_EXAMPLE_HPP
#define MINU_AUGLANG_LINEAR_EXAMPLE_HPP
/**
 * Solution of the burgers equation using the augmented Langrangian mathod
 * assuming a guess has been generated for the initial conditions
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
      namespace burgers
      {
        namespace inverse // min time optimal control
        {
          namespace auglang // Augmented Langrangian
          {

            /**
             * Radau example using the burgers PDE 
             * Using the dual number approach to calcuate the sensitivities
             */
            using Slice = torch::indexing::Slice;
            // Global parameters with default values
            torch::Tensor xf, xm1, xNp1;
            double nu = 0.1, h = 0.01;
            int MaxNbrStep = 10000; // Limit the number of steps to avoid long running times
 

            void set_xf(const torch::Tensor &xf)
            {
              x1f = xf;
            }

            void set_xm1(const torch::Tensor &xm1v)
            {
              xm1 = xm1v;
            }

            void set_xNp1(const torch::Tensor &xNp1v)
            {
              xNp1 = xNp1v;
            }

            void set_nu(double nu_val)
            {
              nu = nu_val;
            }

            void set_h(double h_val)
            {
              h = h_val;
            }



            /**
             * Dynamics calculated for the augmented Langrangian formulation
             */
            TensorDual lineardyns_Lang(const TensorDual &t,
                                       const TensorDual &y,
                                       const TensorDual &params)
            {
              int N= y.r.size(1);
              auto dudts = {};
              TensorDual um1, u, up1;
              for ( int i=0; i < N;i++)
              {
                if ( i==0)
                {
                   um1 = xm1; 
                }
                else 
                {
                  um1 = y.index({Slice(), Slice(i-1, i)});
                }
                auto un = y.index({Slice(), Slice(i, i+1)});
                if ( i==N-1)
                {
                   up1 = xNp1; 
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                dudt = -un*(un-um1)/h + nu*(up1-2*un+um1)/h/h;
                dudts.push_back(dudt);
              }

              auto real_dyns = TensorDual::cat(dudts);

              return real_dyns;
            }


            TensorMatDual jac_Lang(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              int N= y.r.size(1);
              int D = y.d.size(2);
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), N, N}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), N, N, D}, torch::kFloat64));
              auto dudts = {};
              TensorDual um1, u, up1;
              for ( int i=0; i < N;i++)
              {
                if ( i==0)
                {
                   um1 = xm1; 
                }
                else 
                {
                  um1 = y.index({Slice(), Slice(i-1, i)});
                }
                auto un = y.index({Slice(), Slice(i, i+1)});
                if ( i==N-1)
                {
                   up1 = xNp1; 
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                //dudt = -un*(un-um1)/h + nu*(up1-2*un+um1)/h/h;
                if ( i == 0)
                {
                  jac.index_put_({Slice(), 0, 0}, -2*un/h-2* nu/h/h);
                  jac.index_put_({Slice(), 0, 1}, nu/h/h);
                }
                else if ( i == N-1)
                {
                  jac.index_put_({Slice(), N-1, N-2}, un/h/h);
                  jac.index_put_({Slice(), N-1, N-1}, -2*un/h-2* nu/h/h);
                }
                else
                {
                  jac.index_put_({Slice(), i, i-1}, un/h+nu/h/h);
                  jac.index_put_({Slice(), i, i}, -2*un/h-2* nu/h/h);
                  jac.index_put_({Slice(), i, i+1}, nu/h/h);
                }
              }
              
              return jac;
            }





            /**
             * Radau example using a stiff Burgers equation
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
                                               const torch::Tensor &ftt,
                                               const torch::Tensor &lambdap,
                                               const torch::Tensor &mup,           
                                               const torch::Tensor &params)
            {
              std::cerr << "Starting the augmented Langrangian calculation";
              std::cerr << "xic=" << xic;
              std::cerr << "x=" << x;
              std::cerr << "lambdap=" << lambdap;
              std::cerr << "mup=" << mup;
              std::cerr << "params=" << params;
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
              auto ft = TensorDual(ftt.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
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
              
              
              auto errors = torch::cat({c1x.r}, 1);
              auto error_norm = torch::cat({c1x.r}, 1).norm();
              //The jacobian is block diagonal  
              auto jac = torch::zeros({M, 1, 1}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, c1x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1

           
              return std::make_tuple(f.r, grads, errors, error_norm, jac);
            }


          }
        }
      }
    }
  }

} // namespace janus

#endif