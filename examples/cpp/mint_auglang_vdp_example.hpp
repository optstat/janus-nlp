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
            double umin = 0.01;
            double umax = 10.0;
            double mu = 1.0;
            double W = 0.01;

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

            void set_mu(double muval)
            {
              mu = muval;
            }

            void set_W(double Wval)
            {
              W = Wval;
            }

          TensorDual calc_control(const TensorDual &p2)
          {
            auto ustar = -p2 / W;
            return ustar;
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
              auto ustar = calc_control(p2);

              auto dx1dt = x2;
              auto dx2dt = mu * ((1 - x1 * x1) * x2) - x1 + ustar;
              auto dp1dt = p2 * mu * (-2 * x1) * x2 - p2;
              auto dp2dt = p1 + p2 * mu * (1 - x1 * x1);
              auto dLdt = 0.5 * W * ustar * ustar;

              auto real_dyns = TensorDual::cat({dp1dt, dp2dt, dx1dt, dx2dt, dLdt});

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
              auto ustar = calc_control(p2);

              // auto p1 = p.index({Slice(), Slice(0,1)});

              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 5, 5}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 5, 5, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);

              jac.index_put_({Slice(), Slice(0, 1), 1}, mu * (-2 * x1) * x2 - 1.0);
              jac.index_put_({Slice(), Slice(0, 1), 2}, -p2 * mu * 2 * x2);
              jac.index_put_({Slice(), Slice(0, 1), 3}, -p2 * mu * 2 * x1);
              // p1 + p2 * ustar * (1 - x1 * x1)
              jac.index_put_({Slice(), Slice(0, 1), 0}, one);
              jac.index_put_({Slice(), Slice(0, 1), 1}, mu * (1 - x1 * x1));
              jac.index_put_({Slice(), Slice(0, 1), 2}, p2 * mu * (-2 * x1));
              // x2;
              jac.index_put_({Slice(), Slice(2, 3), 3}, one);
              // ustar*((1-x1*x1)*x2)-x1
              jac.index_put_({Slice(), Slice(3, 4), 2}, mu * (-2 * x1 * x2) - 1.0);
              jac.index_put_({Slice(), Slice(3, 4), 3}, mu * ((1 - x1 * x1)));

              // 0.5*W*ustar*ustar
              // ustar = -p2/W
              jac.index_put_({Slice(), Slice(1, 2), 1}, -1.0 / W);

              return jac;
            }

            torch::Tensor dyns_state(const torch::Tensor &t,
                                     const torch::Tensor &y,
                                     const torch::Tensor &params)
            {
              auto dydt = torch::zeros_like(y);
              auto x1 = y.index({Slice(), Slice(0, 1)});
              auto x2 = y.index({Slice(), Slice(1, 2)});
              auto mu = y.index({Slice(), Slice(2, 3)});
              auto dx1dt = x2;
              auto dx2dt = mu * (1 - x1 * x1) * x2 - x1;
              dydt.index_put_({Slice(), Slice(0, 1)}, dx1dt);
              dydt.index_put_({Slice(), Slice(1, 2)}, dx2dt);
              return dydt;
            }

            torch::Tensor jac_state(const torch::Tensor &t,
                                    const torch::Tensor &y,
                                    const torch::Tensor &params)
            {
              int M = y.size(0);
              int N = 3; // Length of the state space vector in order [x1, x2]
              auto jac = torch::zeros({M, N, N}, torch::kFloat64);
              // x2
              auto x1 = y.index({Slice(), 0});
              auto x2 = y.index({Slice(), 1});
              auto mu = y.index({Slice(), 2});
              jac.index_put_({Slice(), 0, 1}, 1.0);
              // mu*(1-x1*x1)*x2
              jac.index_put_({Slice(), 1, 0}, -2.0 * mu * x1 * x2);
              jac.index_put_({Slice(), 1, 1}, mu * (1 - x1 * x1));
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
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mint_auglangr_propagate(const torch::Tensor &x,
                                                                             const torch::Tensor &lambdap,
                                                                             const torch::Tensor &mup,           
                                                                             const torch::Tensor &params)
            {
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 5; // Length of the state space vector in order [p1, p2, x1, x2]
              int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
              double rtol = 1e-12;
              double atol = 1e-16;
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

              auto p20 = TensorDual(x.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              p20.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto ustar = calc_control(p20);
              //H=p1*x2+p2*ustar*(1-x1*x1)*x2-p2*x1+0.5*W*ustar*ustar+1;
              auto p10 = (p20*ustar*(1-x10*x10)*x20-p20*x10+0.5*W*ustar*ustar+1)/x20;
              auto ft = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 4}, 1.0); // Set the dependency to itself

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

              y.index_put_({Slice(), Slice(0, 1)}, p10);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, p20);       // p2
              y.index_put_({Slice(), Slice(2, 3)}, one*x10); // x1
              y.index_put_({Slice(), Slice(3, 4)}, one*x20); // x2
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

              auto pf = r.y.r.index({Slice(), Slice(0, 2)});
              auto xf = r.y.r.index({Slice(), Slice(2, 4)});
              auto p0 = y0.r.index({Slice(), Slice(0, 2)});
              // Now calculate the boundary conditionsx
              auto x1delta = (r.y.index({Slice(), Slice(2, 3)}) - x1f);
              auto x2delta = (r.y.index({Slice(), Slice(3, 4)}) - x2f);
              auto deltas = torch::cat({x1delta.r, x2delta.r}, 1);  //Boundary conditions
              auto LI = r.y.r.index({Slice(), Slice(4, 5)});

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = ft + LI - lambdap.index({Slice(), 0}) * x1delta -
                  lambdap.index({Slice(), 1}) * x2delta  +
                  0.5 * mup * (x1delta * x1delta + x2delta * x2delta);
              std::cerr << "f = " << f << std::endl;
              janus::print_dual(f);
              // The hamiltonian is zero at the terminal time
              // because this is a minimum time problem
              auto grads = torch::zeros_like(x);
              grads.index_put_({Slice(), Slice(0, 1)}, f.d.index({Slice(), 0, Slice(1, 2)})); // p2
              grads.index_put_({Slice(), Slice(1, 2)}, f.d.index({Slice(), 0, Slice(4, 5)})); // ft
              auto res = std::make_tuple(f.r, deltas, grads);
              return res;
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
              int N = 5; // Length of the state space vector in order [p1, p2, x1, x2]
              int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
              double rtol = 1e-12;
              double atol = 1e-16;
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

              auto p20 = TensorDual(x.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              p20.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto ustar = calc_control(p20);
              //H=p1*x2+p2*ustar*(1-x1*x1)*x2-p2*x1+0.5*W*ustar*ustar+1;
              auto p10 = (p20*ustar*(1-x10*x10)*x20-p20*x10+0.5*W*ustar*ustar+1)/x20;
              auto ft = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 4}, 1.0); // Set the dependency to itself

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

              y.index_put_({Slice(), Slice(0, 1)}, p10);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, p20);       // p2
              y.index_put_({Slice(), Slice(2, 3)}, one*x10); // x1
              y.index_put_({Slice(), Slice(3, 4)}, one*x20); // x2
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

              auto pf = r.y.r.index({Slice(), Slice(0, 2)});
              auto xf = r.y.r.index({Slice(), Slice(2, 4)});
              auto p0 = y0.r.index({Slice(), Slice(0, 2)});
              // Now calculate the boundary conditionsx
              auto x1delta = (r.y.index({Slice(), Slice(2, 3)}) - x1f);
              auto x2delta = (r.y.index({Slice(), Slice(3, 4)}) - x2f);
              //std::cerr << "x2delta=";
              //janus::print_dual(x2delta);
              auto deltas = torch::cat({x1delta.r, x2delta.r}, 1);  //Boundary conditions
              auto LI = r.y.index({Slice(), Slice(4, 5)});

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = ft+LI;
              std::cerr << "f = " << f << std::endl;
              janus::print_dual(f);
              auto grads = torch::zeros_like(x);
              grads.index_put_({Slice(), Slice(0, 1)}, f.d.index({Slice(), 0, Slice(1, 2)})); // p2
              grads.index_put_({Slice(), Slice(1, 2)}, f.d.index({Slice(), 0, Slice(4, 5)})); // ft
              // The hamiltonian is zero at the terminal time
              // because this is a minimum time problem
              auto jac = torch::zeros({M, 2, 2}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, x1delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, x1delta.d.index({Slice(), 0, Slice(4, 5)})); // ft
              jac.index_put_({Slice(), Slice(1, 2), Slice(0, 1)}, x2delta.d.index({Slice(), 0, Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(1, 2), Slice(1, 2)}, x2delta.d.index({Slice(), 0, Slice(4, 5)})); // ft

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