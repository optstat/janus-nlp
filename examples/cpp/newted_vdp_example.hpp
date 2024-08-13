#ifndef NEWTED_VDP_EXAMPLE_HPP
#define NEWTED_VDP_EXAMPLE_HPP
/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/autograd.h>
#include <janus/radauted.hpp>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>
#include "../../src/cpp/lnsrchted.hpp"
#include "../../src/cpp/newtted.hpp"
#include "../../src/cpp/newtte.hpp"
#include "../../src/cpp/lnsrchte.hpp"

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

        /**
         * Radau example using the Van der Pol oscillator
         * Using the Hamiltonian with dual number approach to calcuate the dynamics and
         * the Jacobian
         */
        using Slice = torch::indexing::Slice;
        // Global parameters for simplicity
        double W = 1.0;
        double mu = 4.0;
        double alpha = 1.0;
        double u1min = -2.0;
        double u1max = 2.0;
        torch::Tensor x1f = torch::tensor({-1.0}, {torch::kFloat64});
        torch::Tensor x2f = torch::tensor({-1.0}, {torch::kFloat64});
        torch::Tensor x10 = torch::tensor({1.0}, {torch::kFloat64});
        torch::Tensor x20 = torch::tensor({1.0}, {torch::kFloat64});
        int MaxNbrStep = 100000;
        double umin = 0.01;
        double umax = 10.0;
        // Guesses for the initial values of the Lagrange multipliers
        /**
         * In this problem setup the Hamiltonian has form
         * H' = p1'*x2' + p2'*u*((1-x1*x1)*x2'-x1/alpha)+W*u*u/2+1
         * Where alpha is a scaling factor that scales the original p1, p2 and x2
         * in the  original Hamiltonian
         * H = p1*x2 + p2*u*((1-x1*x1)*x2-x1)+W*u*u/2+1
         * where p1 = p1'/alpha
         *       p2 = p2'/alpha
         *       x2 = x2'*alpha
         *       x1 = x1'
         */

        void set_xf(const torch::Tensor& x1, torch::Tensor& x2)
        {
          x1f = x1;
          x2f = x2;
        }

        void set_x0(const torch::Tensor& x1, torch::Tensor& x2)
        {
          x10 = x1;
          x20 = x2;
        }

        void set_mu(const double& muval)
        {
          mu = muval;
        }


        torch::Tensor slog(const torch::Tensor &x)
        {
          auto sign_x = torch::sign(x);
          auto log_abs_x = torch::log(torch::abs(x) + 1);
          return sign_x * log_abs_x;
        }

        torch::Tensor sloginv(const torch::Tensor &x)
        {
          auto sign_x = torch::sign(x);
          auto exp_abs_x = torch::exp(torch::abs(x)) - 1;
          return sign_x * exp_abs_x;
        }

        std::tuple<torch::Tensor, torch::Tensor> calc_control(const torch::Tensor& x, torch::Tensor& p)
        {
            auto p1 = p.index({Slice(), Slice(0, 1)});
            auto p2 = p.index({Slice(), Slice(1, 2)});
            auto x1 = x.index({Slice(), Slice(0, 1)});
            auto x2 = x.index({Slice(), Slice(1, 2)});
            // We have to solve
            //auto p2 = p2p.sloginv();
            auto u1star = -p2/W;
            auto m1 = u1star < u1min;
            auto m2 = u1star > u1max;
            if ( m1.any().item<bool>())
            {
              u1star.index_put_({m1}, u1min);
            }
            if ( m2.any().item<bool>())
            {
              u1star.index_put_({m2}, u1max);
            }
            auto u2star = (-p2*(1-x1*x1)*x2)/W;
            auto m = u2star < umin;
            if ( m.any().item<bool>())
            {
              u2star.index_put_({m}, umin);
            }
            m = u2star > umax;
            if ( m.any().item<bool>())
            {
              u2star.index_put_({m}, umax);
            }
            auto nan_mask = torch::isnan(u2star);
            auto inf_mask = torch::isinf(u2star);
            auto comb_mask = nan_mask | inf_mask;
            
            if ( comb_mask.any().item<bool>())
            {
              auto one = torch::ones_like(p1.index({comb_mask}));
              u2star.index_put_({comb_mask}, one*1.0e6);
            }
            
            
            //std::cerr << "Control = " << res << std::endl;
            return std::make_tuple(u1star, u2star);
        }

        std::tuple<TensorDual, TensorDual> calc_control_dual(const TensorDual& x, TensorDual& p)
        {
            auto p1 = p.index({Slice(), Slice(0, 1)});
            auto p2 = p.index({Slice(), Slice(1, 2)});
            auto x1 = x.index({Slice(), Slice(0, 1)});
            auto x2 = x.index({Slice(), Slice(1, 2)});
            // We have to solve
            //auto p2 = p2p.sloginv();
            auto u1star = -p2/W;
            auto m1 = u1star < u1min;
            auto m2 = u1star > u1max;
            if ( m1.any().item<bool>())
            {
              auto one = TensorDual::ones_like(p1.index(m1));
              u1star.index_put_({m1}, u1min*one);
            }
            if ( m2.any().item<bool>())
            {
              auto one = TensorDual::ones_like(p1.index(m2));
              u1star.index_put_({m2}, u1max*one);
            }

            auto u2star = (-p2*((1-x1*x1)*x2))/W;
            //auto u2star = TensorDual::ones_like(p1)*mu;
            auto m = u2star < umin;
            if ( m.any().item<bool>())
            {
              u2star.index_put_({m}, umin);
            }
            m = u2star > umax;
            if ( m.any().item<bool>())
            {
              u2star.index_put_({m}, umax);
            }
            auto nan_mask = torch::isnan(u2star.r);
            auto inf_mask = torch::isinf(u2star.r);
            auto comb_mask = nan_mask | inf_mask;
            
            if ( comb_mask.any().item<bool>())
            {
              auto one = TensorDual::ones_like(p1.index(comb_mask));
              u2star.index_put_({comb_mask}, one*1.0e6);
            }

            //std::cerr << "Control = " << res << std::endl;
            return std::make_tuple(u1star, u2star);
        }




        torch::Tensor hamiltonian(const torch::Tensor &x,
                                  const torch::Tensor &p,
                                  double W)
        {
          auto y = torch::cat({p, x}, 1);
          auto p1 = y.index({Slice(), Slice(0, 1)});
          auto p2 = y.index({Slice(), Slice(1, 2)});
          auto x1 = y.index({Slice(), Slice(2, 3)});
          auto x2 = y.index({Slice(), Slice(3, 4)});
          //Remove the control from the computational graph
          auto xc = x.clone().detach().requires_grad_(false);
          auto pc = p.clone().detach().requires_grad_(false);
          auto [u1, u2] = calc_control(xc, pc);
          //std::cerr << "Control = " << u << std::endl;

          //auto p1 = -(p2 *u*((1 - x1 * x1) * x2 - x1 / alpha) + W * u * u / 2 + 1.0/alpha)/x2;
          //This is a minimum time Hamiltonian
          auto H = p1*x2+p2*(u2*((1 - x1 * x1) * x2) - x1+u1)+0.5*W*(u1*u1+u2*u2)+ 1.0;
          //std::cerr << "H=" << H << std::endl;
          //std::cerr << "p1=" << p1 << std::endl;
          //std::cerr << "p2=" << p2 << std::endl;
          //std::cerr << "x1=" << x1 << std::endl;
          //std::cerr << "x2=" << x2 << std::endl;
          return H;
        }



        /**
         *Wrapper function for the Hamiltonian
         */
        torch::Tensor ham(const torch::Tensor &y,
                                  double W)
        {
          torch::Tensor pp = y.index({Slice(), Slice(0, 2)});
          torch::Tensor x = y.index({Slice(), Slice(2, 4)});
          return hamiltonian(x, pp, W);
        }
        


        /**
         * Dynamics calculated according the hamiltonian method
         */
        TensorDual vdpdyns_ham(const TensorDual &t,
                               const TensorDual &y,
                               const TensorDual &params)
        {
          //auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
          auto x = y.index({Slice(), Slice(2, 4)});
          auto p = y.index({Slice(), Slice(0, 2)});
          auto [u1star, u2star] = calc_control(x.r, p.r);
          auto x2 = x.index({Slice(), Slice(1,2)});
          auto x1 = x.index({Slice(), Slice(0,1)});
          auto p2 = p.index({Slice(), Slice(1,2)});
          auto p1 = p.index({Slice(), Slice(0,1)});
          //p1*x2+p2*(u*((1 - x1 * x1) * x2) - x1)+0.5*W*u*u+ 1.0
          auto dx1dt = x2;
          auto dx2dt = u2star*((1-x1*x1)*x2)-x1+u1star;
          auto dp1dt = p2*u2star*(-2*x1*x2-1);
          auto dp2dt = p1+p2*u2star*(1-x1*x1);
          auto real_dyns = TensorDual::cat({dp1dt, dp2dt, dx1dt, dx2dt});


          return real_dyns;
        }
        



        TensorMatDual jac_ham(const TensorDual &t,
                              const TensorDual &y,
                              const TensorDual &params)
        {
          //auto jacv = evalJacDual<double>(y, W, hamiltonian);
          //return jacv;
          auto x = y.index({Slice(), Slice(2, 4)});
          auto p = y.index({Slice(), Slice(0, 2)});
          auto [u1star, u2star] = calc_control(x.r, p.r);
          auto x2 = x.index({Slice(), Slice(1,2)});
          auto x1 = x.index({Slice(), Slice(0,1)});
          auto p2 = p.index({Slice(), Slice(1,2)});
          auto p1 = p.index({Slice(), Slice(0,1)});
          auto jac = TensorMatDual(torch::zeros({y.r.size(0), 4, 4}, torch::kFloat64),
                                    torch::zeros({y.r.size(0), 4, 4, y.d.size(2)}, torch::kFloat64));
          auto one = TensorDual::ones_like(x1);
          //p2*u2star*(-2*x1*x2-1);
          jac.index_put_({Slice(), Slice(0,1), 1}, u2star*((-2*x1*x2)-1.0));
          jac.index_put_({Slice(), Slice(0,1), 2}, -p2*u2star*2*x2);
          jac.index_put_({Slice(), Slice(0,1), 3}, -p2*u2star*2*x1);
          //p1+p2*u2star*(1-x1*x1);
          jac.index_put_({Slice(), Slice(0, 1),0}, one);
          jac.index_put_({Slice(), Slice(0, 1),1}, u2star*(1-x1*x1));
          jac.index_put_({Slice(), Slice(0, 1),2}, p2*u2star*(-2*x1));  
          //x2;
          jac.index_put_({Slice(), Slice(2,3), 3}, one);
          //u2star*((1-x1*x1)*x2)-x1+u1star
          jac.index_put_({Slice(), Slice(3,4), 2}, u2star*(-2*x1*x2)-1.0);
          jac.index_put_({Slice(), Slice(3,4), 3}, u2star*((1-x1*x1))); 
          return jac;
        }
        
        /**
         * The first costate is depandent on the second
         */
        torch::Tensor calc_p10(const torch::Tensor &p20)
        {
          auto p10 = torch::zeros_like(p20);//The value of this is irrelevant
          auto x10t = torch::ones_like(p20)*x10;
          auto x20t = torch::ones_like(p20)*x20;
          auto y = torch::cat({p10, p20, x10t, x20t}, 1);
          auto p = torch::cat({p10, p20}, 1);
          auto x = torch::cat({x10t, x20t}, 1);
          auto [u1star, u2star] = calc_control(x, p);
          std::cerr << "u1star=" << u1star << std::endl;
          std::cerr << "u2star=" << u2star << std::endl;
          std::cerr << "x10t=" << x10t << std::endl;
          std::cerr << "x20t=" << x20t << std::endl;
          std::cerr << "mu=" << mu << std::endl;
          //auto H = p1*x2+p2*(u*((1 - x1 * x1) * x2) - x1)+0.5*W*u*u+ 1.0;
          p10 = -(p20*(u2star*((1 - x10t * x10t) * x20t) - x10t+u1star)+
                  0.5*W*(u1star*u1star+u2star*u2star)+
                  1.0
                 )/x20t;
          //We have to account for situations where p10 is very large because x20t is very small
          auto m_nan =torch::isnan(p10) | torch::isinf(p10);
          if ( m_nan.any().item<bool>())
          {
            auto one = torch::ones_like(p10.index({m_nan}));
            p10.index_put_({m_nan}, 1.0e6*one);
          }
          //std::cerr << "p10p_sloginv=" << p10p_sloginv << std::endl;
          //auto res = slog(p10p_sloginv);
          //std::cerr << "p10=" << res << std::endl;
          return p10;
        }


        TensorDual calc_p10(const TensorDual &p20)
        {
          auto p10 = TensorDual::zeros_like(p20);//The value of this is irrelevant
          auto x10t = TensorDual::ones_like(p20)*x10;
          auto x20t = TensorDual::ones_like(p20)*x20;
          auto y = TensorDual::cat({p10, p20, x10t, x20t});
          auto p = TensorDual::cat({p10, p20});
          auto x = TensorDual::cat({x10t, x20t});
          //We don't need the sensitivities of the control just the real part
          auto [u1star,u2star] = calc_control(x.r, p.r);
          p10 = -(p20*(u2star*((1 - x10t * x10t) * x20t) - x10t+u1star)+
                  0.5*W*(u1star*u1star+u2star*u2star)+
                  1.0
                 )/x20t;

          //std::cerr << "p10p_sloginv=" << p10p_sloginv << std::endl;
          //auto res = slog(p10p_sloginv);
          //std::cerr << "p10=" << res << std::endl;
          return p10;
        }




        /**
         * Radau example using the Van der Pol oscillator
         * using dual numbers for sensitivities
         * The function returns the residuals of the expected
         * end state wrt x1f x2f and final Hamiltonian value
         * using p20 and tf as the input variables (x)
         * The relationship is defined by the necessary conditions
         * of optimality as defined by the Variational approach to
         * optimal control
         */
        torch::Tensor propagate(const torch::Tensor &x, 
                                const torch::Tensor &params)
        {
          double rtol = 1e-3;
          double atol = 1e-6;
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
          // set the device
          // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
          int M = x.size(0); // Number of samples
          int N = 4;         // Number of variables in the ODE
          int D = 1;         // Length of the dual vector keep it small to speed up the calculation since we don't need
                     // dual numbers here but to avoid writing two different methods for dynamics and jacobian
          auto device = x.device();

          TensorDual y = TensorDual(torch::zeros({M, N}, torch::kF64).to(device),
                                    torch::zeros({M, N, D}, torch::kF64).to(device));
          auto p20 = x.index({Slice(), Slice(0, 1)});
          auto p10 = calc_p10(p20);  //This is a dependent variable
          auto ft  = x.index({Slice(), Slice(1, 2)});
          y.r.index_put_({Slice(), Slice(0, 1)}, p10); // p1p
          y.r.index_put_({Slice(), Slice(1, 2)}, p20); // p2p
          y.r.index_put_({Slice(), Slice(2, 3)}, x10); // x1
          y.r.index_put_({Slice(), Slice(3, 4)}, x20); // x2
          y.d.index_put_({Slice(), Slice(0,1), Slice(0,1)}, 1.0);
          y.d.index_put_({Slice(), Slice(1,2), Slice(1,2)}, 1.0);
          y.d.index_put_({Slice(), Slice(2,3), Slice(2,3)}, 1.0);
          y.d.index_put_({Slice(), Slice(3,4), Slice(3,4)}, 1.0);


          auto y0 = y.clone(); //Keep a copy of the initial state                       

          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device),
                                        torch::zeros({M, 2, D}, torch::kFloat64).to(device));
          tspan.r.index_put_({Slice(), Slice(0, 1)}, 0.0);
          // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
          tspan.r.index_put_({Slice(), Slice(1, 2)}, ft);
          tspan.d.index_put_({Slice(), 1, -1}, 1.0); // Sensitivity to the final time

          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          //*options.EventsFcn = vdpEvents;
          // Give the ODE integrator modest tolerances and a maximum number of steps
          // to ensure that reasonable execution times
          auto t0 = tspan.clone();
          options.RelTol = torch::tensor({rtol}, torch::kFloat64).to(device);
          options.AbsTol = torch::tensor({atol}, torch::kFloat64).to(device);
          options.MaxNbrStep = MaxNbrStep;

          // Create l values this is a C++ requirement for non constant references
          auto tspanc = tspan.clone();
          auto paramsc = y*0.0;

          janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, y, options, paramsc); // Pass the correct arguments to the constructor`
          int rescode = r.solve();
          // std::cerr << "r.Last=";
          // std::cerr << r.Last << "\n";
          // std::cerr << "r.h=";
          // janus::print_dual(r.h);

          if (rescode != 0)
          {
            std::cerr << "propagation failed\n";
            // Return a large result to make sure the solver does not fail
            return torch::ones_like(x) * 1.0e6;
          }
          auto pf = r.y.index({Slice(), Slice(0, 2)});
          auto xf = r.y.index({Slice(), Slice(2, 4)});
          auto p0 = y0.index({Slice(), Slice(0, 2)});
          auto x0 = y0.index({Slice(), Slice(2, 4)});
          // Now calculate the boundary conditionsx
          auto x1delta = r.y.index({Slice(), Slice(2, 3)}) - x1f;
          auto x2delta = r.y.index({Slice(), Slice(3, 4)}) - x2f;
          auto Hf = hamiltonian(xf.r, pf.r, W);
          torch::Tensor res = x * 0.0;
          // The hamiltonian is zero at the terminal time
          // because this is a minimum time problem
          res.index_put_({Slice(), Slice(0, 1)}, x1delta.r);
          res.index_put_({Slice(), Slice(1, 2)}, x2delta.r);
          res.index_put_({Slice(), Slice(2, 3)}, Hf*0.0);
          std::cerr << "For input x=";  
          janus::print_tensor(x);
          std::cerr << "Final point=";
          janus::print_tensor(xf.r);
          std::cerr << "propagation result (delta)=";
          janus::print_tensor(res);
          return res;
        }


        /**
         * Radau example using the Van der Pol oscillator
         * using dual numbers for sensitivities
         * The function returns the residuals of the expected
         * end state wrt x1f x2f and final Hamiltonian value
         * using p20 and tf as the input variables (x)
         * The relationship is defined by the necessary conditions
         * of optimality as defined by the Variational approach to
         * optimal control
         */
        torch::Tensor propagate_traj(const torch::Tensor &x, const torch::Tensor &params)
        {

          // set the device
          // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
          int M = x.size(0); // Number of samples
          int N = 4;         // Number of variables in the ODE
          int D = 5;         // Length of the dual vector keep it small to speed up the calculation since we don't need
                     // dual numbers here but to avoid writing two different methods for dynamics and jacobian
          auto device = x.device();

          TensorDual y = TensorDual(torch::zeros({M, N}, torch::kF64).to(device),
                                    torch::zeros({M, N, D}, torch::kF64).to(device));
          auto p10 = x.index({Slice(), Slice(0, 1)});
          auto p20 = x.index({Slice(), Slice(1, 2)});
          auto ft  = x.index({Slice(), Slice(2, 3)});
          y.r.index_put_({Slice(), Slice(0, 1)}, p10); // p1
          y.r.index_put_({Slice(), Slice(1, 2)}, p20); // p2
          y.r.index_put_({Slice(), Slice(2, 3)}, x10); // x1
          y.r.index_put_({Slice(), Slice(3, 4)}, x20); // x2
          //Apply the fact that the H=0 at all times to get the initial control
          //Recalcuate p10
          //p10 = -(p20 *u0*((1 - x10 * x10) * x20 - x10 / alpha) + W * u0 * u0 / 2 + 1.0/alpha)/x20;

          y.r.index_put_({Slice(), Slice(0, 1)}, p10); // p1        
          y.r.index_put_({Slice(), Slice(1, 2)}, p20); // p2
          y.r.index_put_({Slice(), Slice(2, 3)}, x10); // x1
          y.r.index_put_({Slice(), Slice(3, 4)}, x20); // x2    

          y.d.index_put_({Slice(), Slice(0,1), Slice(0,1)}, 1.0);
          y.d.index_put_({Slice(), Slice(1,2), Slice(1,2)}, 1.0);
          y.d.index_put_({Slice(), Slice(2,3), Slice(2,3)}, 1.0);
          y.d.index_put_({Slice(), Slice(3,4), Slice(3,4)}, 1.0);
          auto y0 = y.clone(); //Keep a copy of the initial state                       

          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device),
                                        torch::zeros({M, 2, D}, torch::kFloat64).to(device));
          tspan.r.index_put_({Slice(), Slice{0, 1}}, 0.0);
          // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
          tspan.r.index_put_({Slice(), Slice(1, 2)}, x.index({Slice(), Slice(2, 3)}));
          tspan.d.index_put_({Slice(), 1, -1}, 1.0); // Sensitivity to the final time
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          //*options.EventsFcn = vdpEvents;
          // Give the ODE integrator modest tolerances and a maximum number of steps
          // to ensure that reasonable execution times
          auto t0 = tspan.clone();
          options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
          options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
          options.MaxNbrStep = MaxNbrStep;

          // Create l values this is a C++ requirement for non constant references
          auto tspanc = tspan.clone();
          auto paramsc = TensorDual(params.clone(), torch::zeros({M, params.size(1), D}));

          janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, y, options, paramsc); // Pass the correct arguments to the constructor`
          int rescode = r.solve();
          // std::cerr << "r.Last=";
          // std::cerr << r.Last << "\n";
          // std::cerr << "r.h=";
          // janus::print_dual(r.h);

          if (rescode != 0)
          {
            std::cerr << "propagation failed\n";
            // Return a large result to make sure the solver does not fail
            return torch::ones_like(x) * 1.0e6;
          }
          auto pf = r.y.index({Slice(), Slice(0, 2)});
          auto xf = r.y.index({Slice(), Slice(2, 4)});
          auto p0 = y0.index({Slice(), Slice(0, 2)});
          auto x0 = y0.index({Slice(), Slice(2, 4)});
          // Now calculate the boundary conditionsx
          auto x1delta = r.y.index({Slice(), Slice(2, 3)}) - x1f;
          auto x2delta = r.y.index({Slice(), Slice(3, 4)}) - x2f;
          return r.yout.r;
        }


        /**
         * Calculate the Jacobian of the propagation function
         * Using dual numbers to calculate the sensitivity
         *
         */
        torch::Tensor jac_eval(const torch::Tensor &x, 
                               const torch::Tensor &params)  
        {
          
          // set the device
          // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
          int M = x.size(0);
          int N = 4; // Length of the state space vector in order [p1, p2, x1, x2]
          int D = 5; // Length of the dual vector in order [p1, p2, x1, x2, tf]
          double rtol = 1e-3;
          double atol = 1e-6;
          if ( params.sizes() == 1)
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
          p20.d.index_put_({Slice(), 0, 1}, 1.0); //This is an independent variable whose sensitivity we are interested in
        
          auto ft  = TensorDual(x.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
          ft.d.index_put_({Slice(), 0, 4}, 1.0); //Set the dependency to itself
          TensorDual y = TensorDual(torch::zeros({M, N}, x.options()),
                                    torch::zeros({M, N, D}, x.options()));
          TensorDual one = TensorDual::ones_like(p20);
          auto p10 = calc_p10(p20);  //This is a dependent variable
          y.index_put_({Slice(), Slice(0, 1)}, p10); // p1
          y.index_put_({Slice(), Slice(1, 2)}, p20); // p2p
          y.index_put_({Slice(), Slice(2, 3)}, one*x10); // x1
          y.index_put_({Slice(), Slice(3, 4)}, one*x20); // x2    
          auto y0  = y.clone(); //Copy of the initial conditions       
                   
          


          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          TensorDual tspan = TensorDual(torch::rand({M, 2}, x.options()), torch::zeros({M, 2, D}, x.options()));
          tspan.r.index_put_({Slice(), Slice(0, 1)}, 0.0);
          // tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
          tspan.r.index_put_({Slice(), Slice(1, 2)}, x.index({Slice(), Slice(1, 2)}));
          tspan.d.index_put_({Slice(), 1, -1}, 1.0); // Sensitivity to the final time
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          // Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
          janus::OptionsTeD options = janus::OptionsTeD(); // Initialize with default options
          options.RelTol = torch::tensor({rtol}, x.options());
          options.AbsTol = torch::tensor({atol}, x.options());

          options.MaxNbrStep = MaxNbrStep;
          auto tspanc = tspan.clone();
          auto yc = y.clone();
          auto paramsc = TensorDual(params.clone(), torch::zeros({M, params.size(1), N}));

          janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
          // Call the solve method of the Radau5 class
          int rescode = r.solve();
          // std::cerr << "r.Last=";
          // std::cerr << r.Last << "\n";
          // std::cerr << "r.h=";
          // janus::print_dual(r.h);
          torch::Tensor jacVal = torch::zeros({M, 2, 2}, torch::kFloat64);

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
          auto x1delta = r.y.index({Slice(), Slice(2, 3)}) - x1f;
          std::cerr << "x1delta=";
          janus::print_dual(x1delta);
          auto x2delta = r.y.index({Slice(), Slice(3, 4)}) - x2f;
          std::cerr << "x2delta=";
          janus::print_dual(x2delta);
          jacVal.index_put_({Slice(), 0, 0}, x1delta.d.index({Slice(), 0, 1}));
          jacVal.index_put_({Slice(), 0, 1}, x1delta.d.index({Slice(), 0, -1}));
          jacVal.index_put_({Slice(), 1, 0}, x2delta.d.index({Slice(), 0, 1}));
          jacVal.index_put_({Slice(), 1, 1}, x2delta.d.index({Slice(), 0, -1}));


          std::cerr << "jacobian result=";
          janus::print_tensor(jacVal);

          return jacVal;
        }


        torch::Tensor vdp_solve(torch::Tensor& x)
        {
          auto ft = x.index({Slice(), Slice(1, 2)});
          auto params = torch::ones({1, 2}, torch::kFloat64).to(x.device());
          params.index_put_({0, 0}, 1.0e-3); //rtol
          params.index_put_({0, 1}, 1.0e-6); //atol
          auto res = propagate(x, params); 
          return Jfunc(res);

        }


        /**
         * The goal of this example is to solve for a minimum time optimal control problem
         * that can become very stiff very suddenly requiring the use of dual number
         * sensitivity and global optimization techniques to solve the problem
         */

        std::tuple<torch::Tensor, torch::Tensor> vdpNewt(torch::Tensor& p20p, 
                                                         torch::Tensor& ft, 
                                                         double rtol=1.0e-3,
                                                         double atol=1.0e-6)
        {
          void (*pt)(const torch::Tensor &) = janus::print_tensor;
          void (*pd)(const TensorDual &) = janus::print_dual;
          void (*pmd)(const TensorMatDual &) = janus::print_dual;
          //Release the gil-this is needed when calling pytorch from python
          py::gil_scoped_release no_gil;

          int M = 1;     // Number of samples
          int N = 2;     // Number of unknonwns
          int D = 5;     // Dimension of the dual numbers [p1, p2, x1, x2, tf]
          // The problem has three unknowns p1 p2 and final time tf
          /*
          -28.8603   1.4887   6.1901*/

          auto device = torch::kCPU;
    
          std::cerr << "p20p=" << p20p << "\n";
          std::cerr << "ft=" << ft << "\n";
          torch::Tensor y0 = torch::cat({p20p, ft}, 1).to(device);
          std::cerr << "y0=" << y0 << "\n";
          
          auto params = torch::ones({M, 2}, torch::kFloat64).to(device);
          params.index_put_({Slice(), Slice(0,1)}, rtol); //rtol
          params.index_put_({Slice(), Slice(1,2)}, atol); //atol
          std::cerr << "params=" << params << "\n";
          auto res = newtTe(y0, params, propagate, jac_eval);
          //auto roots = std::get<0>(res);
          //auto check = std::get<1>(res);
          //std::cerr << "roots=" << roots << "\n";
          //std::cerr << "check=" << check << "\n";
          std::cerr << "Calling propagate with y0=" << y0 << "\n"; 
          std::cerr << "Calling propagate with params=" << params << "\n";
          auto prop_res = propagate(y0, params);
          std::cerr << "prop_res sizes=" << prop_res.sizes() << "\n";
          auto errors = Jfunc(prop_res);
          std::cerr << "errors=" << errors << "\n";

          // Now propagate the final solution to get the final sensitivities

          return std::make_tuple(y0, errors);
        }


        torch::Tensor solve_traj(torch::Tensor p10p,
                                 torch::Tensor p20p, 
                                 torch::Tensor ft)
        {
          void (*pt)(const torch::Tensor &) = janus::print_tensor;
          void (*pd)(const TensorDual &) = janus::print_dual;
          void (*pmd)(const TensorMatDual &) = janus::print_dual;
          //Release the gil-this is needed when calling pytorch from python
          py::gil_scoped_release no_gil;

          int M = 1;     // Number of samples
          int N = 2;     // Number of unknonwns
          int D = N + 1; // Dimension of the dual numbers [p1, p2, tf]
          // The problem has three unknowns p1 p2 and final time tf
          /*
          -28.8603   1.4887   6.1901*/

          auto device = torch::kCPU;

          torch::Tensor y0 = torch::cat({p10p, p20p, ft}, 1).to(device);

          torch::Tensor params = torch::zeros_like(y0);
          // Impose limits on the state space

          //auto res = newtTe(y0, params, propagate, jac_eval);
          //auto roots = std::get<0>(res);
          //auto check = std::get<1>(res);
          //std::cerr << "roots=" << roots << "\n";
          //std::cerr << "check=" << check << "\n";
          auto traj = propagate_traj(y0, params);

          return traj;
        }

      }
    }
  }

} // namespace newted_example

#endif