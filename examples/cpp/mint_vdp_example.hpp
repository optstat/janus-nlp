#ifndef MINT_VDP_EXAMPLE_HPP
#define MINT_VDP_EXAMPLE_HPP
/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/autograd.h>
#include <janus/radauted.hpp>
#include <janus/radaute.hpp>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>

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
          double mu = 10.0;
          double W = 0.1;

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

          torch::Tensor calc_control(const torch::Tensor &p1,
                                       const torch::Tensor &p2,
                                       torch::Tensor &x)
            {
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto x2 = x.index({Slice(), Slice(1, 2)});
              // We have to solve
              // auto p2 = p2p.sloginv();
              // Here the Hamiltonian is
              // H=p1*x2+p2*mu*(1-x1*x1)*x2-p2*x1+p2*u+1+0.5*W*u*u
              auto ustar = -p2 / W;
              return ustar;
            }


            TensorDual calc_control(const TensorDual &p1,
                                    const TensorDual &p2,
                                    const TensorDual &x)
            {
              auto ustar = -p2 / W;
              return ustar;
            }






          torch::Tensor hamiltonian(const torch::Tensor &p,
                                    const torch::Tensor &x)
          {
            auto y = torch::cat({p, x}, 1);
            auto p1 = y.index({Slice(), Slice(0, 1)});
            auto p2 = y.index({Slice(), Slice(1, 2)});
            auto x1 = y.index({Slice(), Slice(2, 3)});
            auto x2 = y.index({Slice(), Slice(3, 4)});
            // Remove the control from the computational graph
            auto xc = x.clone().detach().requires_grad_(false);
            auto pc = p.clone().detach().requires_grad_(false);
            auto p1c = pc.index({Slice(), Slice(0, 1)});
            auto p2c = pc.index({Slice(), Slice(1, 2)});
            auto ustar = calc_control(p1c, p2c, xc);
            //override the value of p1 because there is no dependency
            p1 = (p2 * mu * (1 - x1 * x1) * x2 - p2 * x1+p2*ustar+1+0.5*W*ustar*ustar)/x2;
            auto H = p1 * x2 + p2 * mu * (1 - x1 * x1) * x2 - p2 * x1 + 1+p2*ustar+0.5*W*ustar*ustar;

            return H;
          }

          TensorDual hamiltonian(const TensorDual &p,
                                 const TensorDual &x)
          {
            auto y = TensorDual::cat({p, x});
            auto p1 = y.index({Slice(), Slice(0, 1)});
            auto p2 = y.index({Slice(), Slice(1, 2)});
            auto x1 = y.index({Slice(), Slice(2, 3)});
            auto x2 = y.index({Slice(), Slice(3, 4)});
            auto ustar = calc_control(p1, p2, x); 
            //We need total sensitivity ih the dual formulation
            p1 = (p2 * mu * (1 - x1 * x1) * x2 - p2 * x1+p2*ustar+1+0.5*W*ustar*ustar)/x2;

            auto H = p1 * x2 + p2 * mu * (1 - x1 * x1) * x2 - p2 * x1 + 1+p2*ustar+0.5*W*ustar*ustar;

            return H;
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
             * Dynamics calculated according the hamiltonian method
             */
            torch::Tensor vdpdyns(const torch::Tensor &t,
                                  const torch::Tensor &y,
                                  const torch::Tensor &params)
            {
              // auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
              auto x = y.index({Slice(), Slice(2, 4)});
              auto p = y.index({Slice(), Slice(0, 2)});
              // std::cerr << "Control = " << ustar << std::endl;
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto p2 = p.index({Slice(), Slice(1, 2)});
              auto p1 = p.index({Slice(), Slice(0, 1)});
              auto ustar = calc_control(p1, p2, x);
              // H=p1 * x2 + p2 * mu * (1 - x1 * x1) * x2 - p2 * x1 + 1+p2*u2star
              auto dx1dt = x2;
              auto dx2dt = mu * ((1 - x1 * x1) * x2) - x1 + ustar;
              auto dp1dt = p2 * mu * (-2 * x1) * x2 - p2;
              auto dp2dt = p1 + p2 * mu * (1 - x1 * x1);
              auto real_dyns = torch::cat({dp1dt, dp2dt, dx1dt, dx2dt}, 1);

              return real_dyns;
            }

            /**
             * Dynamics calculated according the hamiltonian method
             */
            TensorDual vdpdyns_dual(const TensorDual &t,
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
              auto ustar = calc_control(p1, p2, x);

              auto dx1dt = x2;
              auto dx2dt = mu * ((1 - x1 * x1) * x2) - x1 + ustar;
              auto dp1dt = p2 * mu * (-2 * x1) * x2 - p2;
              auto dp2dt = p1 + p2 * mu * (1 - x1 * x1);

              auto real_dyns = TensorDual::cat({dp1dt, dp2dt, dx1dt, dx2dt});

              return real_dyns;
            }


            TensorMatDual jac_dual(const TensorDual &t,
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
              auto ustar = calc_control(p1, p2, x);

              // auto p1 = p.index({Slice(), Slice(0,1)});

              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 4, 4}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 4, 4, y.d.size(2)}, torch::kFloat64));
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

              return jac;
            }



          /**
           * Dynamics calculated for the augmented Langrangian formulation
           */
          TensorDual vdpdyns_auglang(const TensorDual &t,
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
            auto ustar = calc_control(p1, p2, x);

            auto dx1dt = x2;
            auto dx2dt = mu * ((1 - x1 * x1) * x2) - x1+ustar;
            auto dp1dt = p2 * mu * (-2 * x1) * x2 - p2;
            auto dp2dt = p1 + p2 * mu * (1 - x1 * x1);
            auto dLdt = 0.5*W*ustar*ustar;

            auto real_dyns = TensorDual::cat({dp1dt, dp2dt, dx1dt, dx2dt, dLdt});

            return real_dyns;
          }


          TensorMatDual jac_auglang(const TensorDual &t,
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
            auto ustar = calc_control(p1, p2, x);

            // auto p1 = p.index({Slice(), Slice(0,1)});

            auto jac = TensorMatDual(torch::zeros({y.r.size(0), 5, 5}, torch::kFloat64),
                                     torch::zeros({y.r.size(0), 5, 5, y.d.size(2)}, torch::kFloat64));
            auto one = TensorDual::ones_like(x1);

            jac.index_put_({Slice(), Slice(0, 1), 1}, mu *(-2 * x1) * x2 - 1.0);
            jac.index_put_({Slice(), Slice(0, 1), 2}, -p2 * mu * 2 * x2);
            jac.index_put_({Slice(), Slice(0, 1), 3}, -p2 * mu * 2 * x1);
            //p1 + p2 * ustar * (1 - x1 * x1)
            jac.index_put_({Slice(), Slice(0, 1), 0}, one);
            jac.index_put_({Slice(), Slice(0, 1), 1}, mu * (1 - x1 * x1));
            jac.index_put_({Slice(), Slice(0, 1), 2}, p2 * mu * (-2 * x1));
            // x2;
            jac.index_put_({Slice(), Slice(2, 3), 3}, one);
            // ustar*((1-x1*x1)*x2)-x1
            jac.index_put_({Slice(), Slice(3, 4), 2}, mu * (-2 * x1 * x2) - 1.0);
            jac.index_put_({Slice(), Slice(3, 4), 3}, mu * ((1 - x1 * x1)));

            //0.5*W*ustar*ustar
            //ustar = -p2/W
            jac.index_put_({Slice(), Slice(1, 2), 1}, -1.0/W);

            return jac;
          }

            /**
             * Radau example using the Van der Pol oscillator
             * without sensitivity calculations
             * The function returns the residuals of the expected
             * end state wrt x1f x2f and final Hamiltonian value
             * using p20 and tf as the input variables (x)
             * The relationship is defined by the necessary conditions
             * of optimality as defined by the Variational approach to
             * optimal control
             */
            torch::Tensor mint_propagate(const torch::Tensor &x,
                                         const torch::Tensor &params)
            {
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 4; // Length of the state space vector in order [p1, p2, x1, x2]
              int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
              double rtol = 1e-12;
              double atol = 1e-16;
              if (params.sizes() == 1)
              {
                rtol = params.index({0}).item<double>();
                atol = params.index({1}).item<double>();
              }
              else
              {
                rtol = params.index({Slice(), 0}).item<double>();
                atol = params.index({Slice(), 1}).item<double>();
              }
              auto p10 = TensorDual(x.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              p10.d.index_put_({Slice(), 0, 0}, 1.0); // This is an independent variable whose sensitivity we are interested in

              auto p20 = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              p20.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in

              auto ft = TensorDual(x.index({Slice(), Slice(2, 3)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 4}, 1.0); // Set the dependency to itself

              TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                        torch::zeros({M, N, D}, x.options()));
              TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

              y.index_put_({Slice(), Slice(0, 1)}, p10);       // p1
              y.index_put_({Slice(), Slice(1, 2)}, p20);       // p2
              y.index_put_({Slice(), Slice(2, 3)}, one * x10); // x1
              y.index_put_({Slice(), Slice(3, 4)}, one * x20); // x2
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
              auto paramsc = TensorDual(params.clone(), torch::zeros({M, params.size(1), N}));

              janus::RadauTeD r(vdpdyns_dual, jac_dual, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
              // Call the solve method of the Radau5 class
              int rescode = r.solve();

              if (rescode != 0)
              {
                std::cerr << "propagation failed\n";
                // Return a large result to make sure the solver does not fail
                return torch::ones_like(x) * 1.0e6;
              }
              auto pf = r.y.r.index({Slice(), Slice(0, 2)});
              auto xf = r.y.r.index({Slice(), Slice(2, 4)});
              auto p0 = y0.r.index({Slice(), Slice(0, 2)});
              // Now calculate the boundary conditionsx
              auto x1delta = (r.y.r.index({Slice(), Slice(2, 3)}) - x1f);
              auto x2delta = (r.y.r.index({Slice(), Slice(3, 4)}) - x2f);
              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              auto Hf = hamiltonian(pf, xf);
              torch::Tensor res = x * 0.0;
              // The hamiltonian is zero at the terminal time
              // because this is a minimum time problem
              res.index_put_({Slice(), Slice(0, 1)}, x1delta);
              res.index_put_({Slice(), Slice(1, 2)}, x2delta);
              res.index_put_({Slice(), Slice(2, 3)}, Hf);
              std::cerr << "For input x=";
              janus::print_tensor(x);
              std::cerr << "Initial point=";
              janus::print_tensor(y0.r);
              std::cerr << "Final point=";
              janus::print_tensor(xf);
              std::cerr << "propagation result (delta)=";
              janus::print_tensor(res);
              return res;
            }

            std::tuple<torch::Tensor, torch::Tensor> propagate_state(const torch::Tensor &mu,
                                                                     const torch::Tensor &x1,
                                                                     const torch::Tensor &x2,
                                                                     const torch::Tensor &ft,
                                                                     const torch::Tensor &params)
            {
              auto rtol = params.index({0}).item<double>();
              auto atol = params.index({1}).item<double>();
              auto M = x1.size(0);
              auto N = 3; // Length of the state space vector in order [x1, x2, mu]
              auto device = x1.device();
              auto type = x1.dtype();
              auto y = torch::zeros({M, N}, type).to(device);
              y.index_put_({Slice(), 0}, x1);
              y.index_put_({Slice(), 1}, x2);
              y.index_put_({Slice(), 2}, mu);
              torch::Tensor tspan = torch::rand({M, 2}, torch::kFloat64).to(device);
              tspan.index_put_({Slice(), Slice(0, 1)}, 0.0);
              // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
              tspan.index_put_({Slice(), Slice(1, 2)}, ft);

              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              janus::OptionsTe options = janus::OptionsTe(); // Initialize with default options
              // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
              //*options.EventsFcn = vdpEvents;
              // Give the ODE integrator modest tolerances and a maximum number of steps
              // to ensure that reasonable execution times
              options.RelTol = torch::tensor({rtol}, torch::kFloat64).to(device);
              options.AbsTol = torch::tensor({atol}, torch::kFloat64).to(device);
              options.MaxNbrStep = MaxNbrStep;

              // Create l values this is a C++ requirement for non constant references
              auto tspanc = tspan.clone();
              auto paramsc = y * 0.0;

              janus::RadauTe r(dyns_state, jac_state, tspan, y, options, paramsc); // Pass the correct arguments to the constructor`
              int rescode = r.solve();
              auto x1f = r.y.index({Slice(), Slice(0, 1)});
              auto x2f = r.y.index({Slice(), Slice(1, 2)});
              return std::make_tuple(x1f, x2f);
            }





          torch::Tensor mint_vdp_solve(torch::Tensor &x)
          {
            auto params = torch::ones({1, 2}, torch::kFloat64).to(x.device());
            params.index_put_({0, 0}, 1.0e-6); // rtol
            params.index_put_({0, 1}, 1.0e-9); // atol
            auto res = mint_propagate(x, params);
            return res;
          }

          /**
           * Calculate the Jacobian of the propagation function
           * Using dual numbers to calculate the sensitivity
           *
           */
          torch::Tensor jac_eval(const torch::Tensor &x,
                                 const torch::Tensor &params)
          {
            auto xc = x.clone();
            // set the device
            // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
            int M = x.size(0);
            int N = 4; // Length of the state space vector in order [p1, p2, x1, x2]
            int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
            double rtol = 1e-12;
            double atol = 1e-16;
            if (params.sizes() == 1)
            {
              rtol = params.index({0}).item<double>();
              atol = params.index({1}).item<double>();
            }
            else
            {
              rtol = params.index({Slice(), 0}).item<double>();
              atol = params.index({Slice(), 1}).item<double>();
            }
            auto p10 = TensorDual(x.index({Slice(), Slice(0, 1)}).clone(), torch::zeros({M, 1, D}, x.options()));
            p10.d.index_put_({Slice(), 0, 0}, 1.0); // This is an independent variable whose sensitivity we are interested in

            auto p20 = TensorDual(x.index({Slice(), Slice(1, 2)}).clone(), torch::zeros({M, 1, D}, x.options()));
            p20.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in

            auto ft = TensorDual(x.index({Slice(), Slice(2, 3)}).clone(), torch::zeros({M, 1, D}, x.options()));
            ft.d.index_put_({Slice(), 0, -1}, 1.0); // Set the dependency to itself

            TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                      torch::zeros({M, N, D}, x.options()));
            TensorDual one = TensorDual(torch::ones({M, 1}, x.options()), torch::zeros({M, 1, D}, x.options()));

            y.index_put_({Slice(), Slice(0, 1)}, p10);       // p1
            y.index_put_({Slice(), Slice(1, 2)}, p20);       // p2p
            y.index_put_({Slice(), Slice(2, 3)}, one * x10); // x1
            y.index_put_({Slice(), Slice(3, 4)}, one * x20); // x2
            auto y0 = y.clone();                             // Copy of the initial conditions

            // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
            TensorDual tspan = TensorDual(torch::zeros({M, 2}, x.options()), torch::zeros({M, 2, D}, x.options()));
            tspan.r.index_put_({Slice(), Slice(0, 1)}, 0.0);
            // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
            tspan.index_put_({Slice(), Slice(1, 2)}, ft);
            // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
            // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
            janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
            options.RelTol = torch::tensor({rtol}, x.options());
            options.AbsTol = torch::tensor({atol}, x.options());

            options.MaxNbrStep = MaxNbrStep;
            auto yc = y.clone();
            auto paramsode = TensorDual::zeros_like(y);

            janus::RadauTeD r(vdpdyns_dual, jac_dual, tspan, y, options, paramsode); // Pass the correct arguments to the constructor
            // Call the solve method of the Radau5 class
            int rescode = r.solve();
            // std::cerr << "r.Last=";
            // std::cerr << r.Last << "\n";
            // std::cerr << "r.h=";
            // janus::print_dual(r.h);
            torch::Tensor jacVal = torch::zeros({M, 3, 3}, torch::kFloat64);

            if (rescode != 0)
            {
              std::cerr << "propagation failed\n";
              // Return default zeros.  These values will not be used
              return jacVal;
            }

            // Extract the final values from the solver
            auto pfp = r.y.index({Slice(), Slice(0, 2)});
            auto xf = r.y.index({Slice(), Slice(2, 4)});
            std::cerr << "xf=";
            janus::print_dual(xf);
            auto x1delta = (r.y.index({Slice(), Slice(2, 3)}) - x1f);
            std::cerr << "x1delta=";
            janus::print_dual(x1delta);
            auto x2delta = (r.y.index({Slice(), Slice(3, 4)}) - x2f);
            std::cerr << "x2delta=";
            janus::print_dual(x2delta);
            //Evaluate the Hamiltonian at the next to final point

            auto Hf = hamiltonian(pfp, xf);
            jacVal.index_put_({Slice(), 0, 0}, x1delta.d.index({Slice(), 0, 0}));
            jacVal.index_put_({Slice(), 0, 1}, x1delta.d.index({Slice(), 0, 1}));
            jacVal.index_put_({Slice(), 0, 2}, x1delta.d.index({Slice(), 0, -1}));

            jacVal.index_put_({Slice(), 1, 0}, x2delta.d.index({Slice(), 0, 0}));
            jacVal.index_put_({Slice(), 1, 1}, x2delta.d.index({Slice(), 0, 1}));
            jacVal.index_put_({Slice(), 1, 2}, x2delta.d.index({Slice(), 0, -1}));
            

            jacVal.index_put_({Slice(), 2, 0}, Hf.d.index({Slice(), 0, 0}));
            jacVal.index_put_({Slice(), 2, 1}, Hf.d.index({Slice(), 0, 1}));
            jacVal.index_put_({Slice(), 2, 2}, Hf.d.index({Slice(), 0, -1}));

            janus::print_tensor(jacVal);

            return jacVal;
          }
            TensorMatDual mint_jac_ham(const TensorDual &t,
                                       const TensorDual &y,
                                       const TensorDual &params)
            {
              // auto jacv = evalJacDual<double>(y, W, hamiltonian);
              // return jacv;
              auto x = y.index({Slice(), Slice(2, 4)});
              auto p = y.index({Slice(), Slice(0, 2)});
              auto p1 = p.index({Slice(), Slice(0, 1)});
              auto p2 = p.index({Slice(), Slice(1, 2)});
              auto ustar = calc_control(p1.r, p2.r, x.r);
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 4, 4}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 4, 4, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);
              // p2*(ustar*(-2*x1*x2)-1);
              jac.index_put_({Slice(), Slice(0, 1), 1}, mu * ((-2 * x1 * x2)) - 1.0);
              jac.index_put_({Slice(), Slice(0, 1), 2}, -p2 * mu * 2 * x2);
              jac.index_put_({Slice(), Slice(0, 1), 3}, -p2 * mu * 2 * x1);
              // p1+p2*ustar*(1-x1*x1);
              jac.index_put_({Slice(), Slice(0, 1), 0}, one);
              jac.index_put_({Slice(), Slice(0, 1), 1}, mu * (1 - x1 * x1));
              jac.index_put_({Slice(), Slice(0, 1), 2}, p2 * mu * (-2 * x1));
              // x2;
              jac.index_put_({Slice(), Slice(2, 3), 3}, one);
              // ustar*((1-x1*x1)*x2)-x1
              jac.index_put_({Slice(), Slice(3, 4), 2}, mu * (-2 * x1 * x2) - 1.0);
              jac.index_put_({Slice(), Slice(3, 4), 3}, mu * ((1 - x1 * x1)));
              return jac;
            }




          /**
           * Calculate the Jacobian of the propagation function using finite differences
           * Using dual numbers to calculate the sensitivity
           *
           */
          torch::Tensor jac_eval_fd(const torch::Tensor &x,
                                 const torch::Tensor &params)
          {
            int M = x.size(0);
            torch::Tensor jacVal = torch::zeros({M, 3, 3}, torch::kFloat64);

            auto p1 = x.index({Slice(), Slice(0, 1)});
            auto hp1 = 1.0e-8*torch::max(torch::abs(p1), torch::ones_like(p1)).reshape_as(p1);
            auto xc = x.clone();
            auto xp1ph = x.clone();
            xp1ph.index_put_({Slice(), Slice(0, 1)}, p1 + hp1);
            auto resp1ph = mint_propagate(xp1ph, params);
            auto xp1mh = x.clone();
            xp1mh.index_put_({Slice(), Slice(0, 1)}, p1 - hp1);
            auto resp1mh = mint_propagate(xp1mh, params);
            auto dp1 = (resp1ph - resp1mh) / (2.0 * hp1);
            jacVal.index_put_({Slice(), Slice(), 0}, dp1);

            auto p2 = x.index({Slice(), Slice(1, 2)});
            auto hp2 = 1.0e-8*torch::max(torch::abs(p2), torch::ones_like(p2)).reshape_as(p2);
            auto xp2ph = x.clone();
            xp2ph.index_put_({Slice(), Slice(1, 2)}, p2 + hp2);
            auto resp2ph = mint_propagate(xp2ph, params);
            auto xp2mh = x.clone();
            xp2mh.index_put_({Slice(), Slice(1, 2)}, p2 - hp2);
            auto resp2mh = mint_propagate(xp2mh, params);
            auto dp2 = (resp2ph - resp2mh) / (2.0 * hp2);
            jacVal.index_put_({Slice(), Slice(), 1}, dp2);

            auto ft = x.index({Slice(), Slice(2, 3)});
            auto hft = 1.0e-8*torch::max(torch::abs(ft), torch::ones_like(ft)).reshape_as(ft);
            auto xftph = x.clone();
            xftph.index_put_({Slice(), Slice(2, 3)}, ft + hft);
            auto resftph = mint_propagate(xftph, params);
            auto xftmh = x.clone();
            xftmh.index_put_({Slice(), Slice(2, 3)}, ft - hft);
            auto resftmh = mint_propagate(xftmh, params);
            auto dft = (resftph - resftmh) / (2.0 * hft);
            jacVal.index_put_({Slice(), Slice(), 2}, dft);

            return jacVal;
          }


          torch::Tensor mint_jac_eval(torch::Tensor &x)
          {
            auto params = torch::ones({1, 2}, torch::kFloat64).to(x.device());
            params.index_put_({0, 0}, 1.0e-6); // rtol
            params.index_put_({0, 1}, 1.0e-9); // atol
            auto jac = jac_eval(x, params);
            return jac;
          }

          torch::Tensor mint_jac_eval_fd(torch::Tensor &x)
          {
            auto params = torch::ones({1, 2}, torch::kFloat64).to(x.device());
            params.index_put_({0, 0}, 1.0e-6); // rtol
            params.index_put_({0, 1}, 1.0e-9); // atol
            auto jac = jac_eval_fd(x, params);
            return jac;
          }

        }
      }
    }
  }

} // namespace janus

#endif