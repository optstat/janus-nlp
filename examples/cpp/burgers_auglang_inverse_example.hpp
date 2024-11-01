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
#include <vector>
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

            torch::Tensor nu, dx;

            /**
             * Radau example using the burgers PDE 
             * Using the dual number approach to calcuate the sensitivities
             */
            using Slice = torch::indexing::Slice;
            // Global parameters with default values
            torch::Tensor xf, xm1, xNp1;
            int MaxNbrStep = 10000; // Limit the number of steps to avoid long running times
 
            TensorDual burgers_dyns_dual(const TensorDual &t,
                                            const TensorDual &y,
                                            const TensorDual &params)
            {
              int M = y.r.size(0);
              int N = y.r.size(1);
              int Nadj = N+2;  //We will need to match the boundary conditions

              std::vector<TensorDual> dudts = {};
              for ( int i=0; i < N; i++)
              {
                TensorDual um1, un, up1;
                if (i==0)
                {
                   um1 = y.index({Slice(), Slice(N-1, N)});
                }
                else 
                {
                  um1 = y.index({Slice(), Slice(i-1,i)});
                }
                if ( i == N-1)
                {
                   up1 = y.index({Slice(), Slice(0,1)});
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                un = y.index({Slice(), Slice(i, i+1)});
                // du_dx[i] = -uc*(up1-um1)/(2*dx) + nu*(up1-2*uc+um1)/dx/dx
                auto dudt = -un*(up1-um1)/(2*dx) + nu*(up1-2*un+um1)/dx/dx;
                dudts.push_back(dudt);
              }
              return TensorDual::cat(dudts);
            }
            

            //Generate a test wrapper for the dynamics
            TensorMatDual burgers_jac_dual(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              int M = y.r.size(0);
              int N= y.r.size(1);
              int D = y.d.size(2);
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), N, N}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), N, N, D}, torch::kFloat64));
              std::vector<TensorDual> dudts = {};
              TensorDual um1, un, up1;
              for ( int i=0; i < N;i++)
              {
                if ( i==0)
                {
                  um1 = y.index({Slice(), Slice(N-1, N)});
                }
                else
                {
                  um1 = y.index({Slice(), Slice(i-1, i)});
                }
                auto un = y.index({Slice(), Slice(i, i+1)});
                if ( i==N-1)
                {
                   up1 = y.index({Slice(), Slice(0, 1)});
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                //-un*(up1-um1)/(2*dx) + nu*(up1-2*un+um1)/dx/dx;
                if ( i == 0)
                {
                  jac.index_put_({Slice(), 0, 0}, -(up1-um1)/dx-2* nu/dx/dx);
                  jac.index_put_({Slice(), 0, 1}, -un/(2*dx)+nu/dx/dx);
                }
                else if ( i == N-1)
                {
                  jac.index_put_({Slice(), N-1, N-2}, un/dx/dx+nu/dx/dx);
                  jac.index_put_({Slice(), N-1, N-1}, -(up1-um1)/dx-2* nu/dx/dx);
                }
                else
                {
                  jac.index_put_({Slice(), i, i-1}, -un*up1/(2*dx)+nu/dx/dx);
                  jac.index_put_({Slice(), i, i},-(up1-um1)/(2*dx) + nu*(up1-2+um1)/dx/dx);
                  jac.index_put_({Slice(), i, i+1}, un/(2*dx)+nu/dx/dx);
                }
              }
              
              return jac;
            }

            std::tuple<torch::Tensor, torch::Tensor> forward_dual(const torch::Tensor& x0, //The initial conditions in a batch tensor
                                                                  const torch::Tensor& ft, //The final time for whole batch
                                                                  const torch::Tensor& nup,
                                                                  const torch::Tensor& dxp,
                                                                  const torch::Tensor& params)
            {
              int M = x0.size(0);
              int N = x0.size(1);
              int D = 1;  //There is only one independent variable
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
              auto y0 = TensorDual(x0.clone(), torch::zeros({1, 2, D}, x0.options()));
              TensorDual tspan = TensorDual(torch::zeros({1, 2}, x0.options()), torch::zeros({1, 2, D}, x0.options()));
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.r.index_put_({Slice(), Slice(1, 2)}, ft);

              janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
              options.RelTol = torch::tensor({rtol}, x0.options());
              options.AbsTol = torch::tensor({atol}, x0.options());

              options.MaxNbrStep = MaxNbrStep;
              auto tspanc = tspan.clone();
              auto yc = y0.clone();
              auto paramsc = TensorDual::zeros_like(y0);
              janus::RadauTeD r(burgers_dyns_dual, burgers_jac_dual, tspan, yc, options, paramsc); // Pass the correct arguments to the constructor
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
              auto res = r.yout;
              return std::make_tuple(res.r, res.d);

            }


            torch::Tensor burgers_dyns(const torch::Tensor &t,
                                    const torch::Tensor &y,
                                    const torch::Tensor &params)
            {
              int M = y.size(0);
              int N = y.size(1);

              std::vector<torch::Tensor> dudts = {};
              for ( int i=0; i < N; i++)
              {
                torch::Tensor um1, un, up1;
                if (i==0)
                {
                   um1 = y.index({Slice(), Slice(N-1, N)});
                }
                else 
                {
                  um1 = y.index({Slice(), Slice(i-1,i)});
                }
                if ( i == N-1)
                {
                   up1 = y.index({Slice(), Slice(0,1)});
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                un = y.index({Slice(), Slice(i, i+1)});
                // du_dx[i] = -uc*(up1-um1)/(2*dx) + nu*(up1-2*uc+um1)/dx/dx
                auto dudt = -un*(up1-um1)/(2*dx) + nu*(up1-2*un+um1)/dx/dx;
                dudts.push_back(dudt);
              }
              return torch::cat(dudts,1);
            }
            

            //Generate a test wrapper for the dynamics
            torch::Tensor burgers_jac(const torch::Tensor &t,
                                   const torch::Tensor &y,
                                   const torch::Tensor &params)
            {
              int M = y.size(0);
              int N= y.size(1);
              auto jac = torch::Tensor(torch::zeros({y.size(0), N, N}, torch::kFloat64));
              std::vector<torch::Tensor> dudts = {};
              torch::Tensor um1, un, up1;
              for ( int i=0; i < N;i++)
              {
                if ( i=0)
                {
                  um1 = y.index({Slice(), Slice(N-1, N)});
                }
                else
                {
                  um1 = y.index({Slice(), Slice(i-1, i)});
                }
                auto un = y.index({Slice(), Slice(i, i+1)});
                if ( i==N-1)
                {
                   up1 = y.index({Slice(), Slice(0, 1)});
                }
                else 
                {
                  up1 = y.index({Slice(), Slice(i+1, i+2)});
                }
                //-un*(up1-um1)/(2*dx) + nu*(up1-2*un+um1)/dx/dx;
                if ( i == 0)
                {
                  jac.index_put_({Slice(), 0, 0}, -(up1-um1)/dx-2* nu/dx/dx);
                  jac.index_put_({Slice(), 0, 1}, -un/(2*dx)+nu/dx/dx);
                }
                else if ( i == N-1)
                {
                  jac.index_put_({Slice(), N-1, N-2}, un/dx/dx+nu/dx/dx);
                  jac.index_put_({Slice(), N-1, N-1}, -(up1-um1)/dx-2* nu/dx/dx);
                }
                else
                {
                  jac.index_put_({Slice(), i, i-1}, -un*up1/(2*dx)+nu/dx/dx);
                  jac.index_put_({Slice(), i, i},-(up1-um1)/(2*dx) + nu*(up1-2+um1)/dx/dx);
                  jac.index_put_({Slice(), i, i+1}, un/(2*dx)+nu/dx/dx);
                }
              }
              
              return jac;
            }


            torch::Tensor forward(const torch::Tensor& x0, //The initial conditions in a batch tensor
                                                                  const torch::Tensor& ft, //The final time for whole batch
                                                                  const torch::Tensor& nup,
                                                                  const torch::Tensor& dxp,
                                                                  const torch::Tensor& params)
            {
              int M = x0.size(0);
              int N = x0.size(1);
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
              torch::Tensor tspan = torch::Tensor(torch::zeros({1, 2}, x0.options()));
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.index_put_({Slice(), Slice(1, 2)}, ft);

              janus::OptionsTe options = janus::OptionsTe(); // Initialize with default options
              options.RelTol = torch::tensor({rtol}, x0.options());
              options.AbsTol = torch::tensor({atol}, x0.options());

              options.MaxNbrStep = MaxNbrStep;
              auto tspanc = tspan.clone();
              auto yc = y0.clone();
              auto paramsc = torch::zeros_like(y0);
              janus::RadauTe r(burgers_dyns, burgers_jac, tspan, yc, options, paramsc); // Pass the correct arguments to the constructor
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
              auto res = r.yout;
              return res;

            }









          }
        }
      }
    }
  }

} // namespace janus

#endif