#ifndef BURGERS_AUGLANG_INVERSE_EXAMPLE_HPP
#define BURGERS_AUGLANG_INVERSE_EXAMPLE_HPP
/**
 * Solution of the burgers equation using the augmented Langrangian mathod
 * assuming a guess has been generated for the initial conditions
 */
#define HEADER_ONLY
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/autograd.h>
#include <janus/radaute.hpp>// Radau5 integrator
#include <janus/radauted.hpp>//Radau integrator with dual number support

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

            double nu; // The viscosity parameter

            /**
             * Radau example using the burgers PDE 
             * Using the dual number approach to calcuate the sensitivities
             */
            using Slice = torch::indexing::Slice;
            // Global parameters with default values
            torch::Tensor xf, xm1, xNp1;
            double nu = 0.1, dx = 0.01;
            int MaxNbrStep = 10000; // Limit the number of steps to avoid long running times
 
            

            //Generate a test wrapper for the dynamics
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
              auto xm1 = y.index({Slice(), Slice(N-1, N)});
              auto xNp1 = y.index({Slice(), Slice(0, 1)});
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

            torch::Tensor burgers_dyns(const torch::Tensor &t,
                                       const torch::Tensor &y,
                                       const torch::Tensor &params)
            {
              int M = y.size(0);
              int N = y.size(1);
              int Nadj = N+2;  //We will need to match the boundary conditions

              std::vector dudts = {};
              for ( int i=0; i < N; i++)
              {
                torch::Tensor um1, un, up1;
                if ( i==0)
                {
                   um1 = y.index({Slice(), N-1});
                }
                else 
                {
                  um1 = y.index({Slice(), i-1});
                }
                if ( i == N-1)
                {
                   up1 = y.index({Slice(), 0});
                }
                else 
                {
                  up1 = y.index({Slice(), i+1});
                }
                auto un = y.index({i});
                auto dudt = -un*(un-um1)/dx + nu*(up1-2*un+um1)/dx/dx;
                dudts.push_back(dudt);
              }
              return torch::cat(dudts,1);
            }

            torch::Tensor burgers_jac(const torch::Tensor &t,
                                      const torch::Tensor &y,
                                      const torch::Tensor &params)
            {
              int M = y.size(0);
              int N = y.size(1);
              int Nadj = N+2;  //We will need to match the boundary conditions
              auto x0m1 = torch.index({Slice(), Slice(N-1:N)}).clone();
              auto x0Np1 = torch.index({Slice(), Slice(0, 1)}).clone();
              auto jac = torch::zeros({M, N, N}, y.options());
              std::vector dudts = {};
              for ( int i=0; i < N; i++)
              {
                torch::Tensor um1, un, up1;
                if ( i==0)
                {
                   um1 = y.index({Slice(), N-1});
                }
                else 
                {
                  um1 = y.index({Slice(), i-1});
                }
                if ( i == N-1)
                {
                   up1 = y.index({Slice(), 0});
                }
                else 
                {
                  up1 = y.index({Slice(), i+1});
                }
                auto un = y.index({i});
                //auto dudt = -un*(un-um1)/dx + nu*(up1-2*un+um1)/dx/dx;
                jac.index_put_({Slice(), i, i-1}, un/dx);
                jac.index_put_({Slice(), i, i}, -2*un/dx-2* nu/dx/dx);
                jac.index_put_({Slice(), i, i+1}, nu/dx/dx);
              }
              return torch::cat(dudts,1);

            }


            torch::Tensor forward(const torch::Tensor& x0, //The initial conditions in a batch tensor
                                  const torch::Tensor& ft, //The final time for whole batch
                                  double nup,
                                  double dxp)
            {

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

              nu = nup;
              dx = dxp;
              auto y0 = x0;
              torch::Tensor tspan = torch::zeros({0, 2}, x0.options());
              tspan.index_put_({Slice(),Slice(0,1)}, 0.0);
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.index_put_({Slice(), Slice(1, 2)}, ft);
              janus::OptionsTe options = janus::OptionsTe(); // Initialize with default options
              options.RelTol = torch::tensor({rtol}, x.options());
              options.AbsTol = torch::tensor({atol}, x.options());

              options.MaxNbrStep = MaxNbrStep;
              auto tspanc = tspan.clone();
              auto yc = y.clone();
              auto paramsc = torch::zeros_like(y);
              janus::RadauTeD r(burgers_dyns, burgers_jac, tspan, yc, options, paramsc); // Pass the correct arguments to the constructor
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
              auto res = r.yout
              return res;

            }






          }
        }
      }
    }
  }

} // namespace janus

#endif