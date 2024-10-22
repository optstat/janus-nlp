#ifndef bratu_ode_included
#define bratu_ode_included
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/autograd.h>
#include <janus/radauted.hpp>
//Create namespaces to avoid creating classes
namespace janus {
  namespace nlp   {
    namespace examples {
      namespace bratu     {
            double lam = 1.0;

            /**
             * Dynamics calculated for the augmented Langrangian formulation
             */
            TensorDual dyns_ext_bratu(const TensorDual &t,
                                 const TensorDual &y,
                                 const TensorDual &params)
            {
              // auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
              auto x1 = y.index({Slice(), Slice(0, 1)});
              auto x2 = y.index({Slice(), Slice(1, 2)});
              auto J = y.index({Slice(), Slice(2, 3)});
              // std::cerr << "Control = " << ustar << std::endl;
              //H = p1*x2-p2*lambda*exp(x1)+1

              auto dx1dt = x2;
              auto dx2dt = -lam*x1.exp();
              auto dJdt = x1*0.0;

              auto dyns = TensorDual::cat({dx1dt, dx2dt, dJdt});

              return dyns;
            }


            TensorMatDual jac_ext_bratu(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              // auto jacv = evalJacDual<double>(y, W, hamiltonian);
              // return jacv;
              auto x2 = y.index({Slice(), Slice(1, 2)});
              auto x1 = y.index({Slice(), Slice(0, 1)});
  
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 2, 2}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 2, 2, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);
              //-p2*l
              jac.index_put_({Slice(), Slice(0, 1), 1}, -lambda*x1.exp());
              jac.index_put_({Slice(), Slice(0, 1), 3}, -lambda*x1.exp());
              // p1;
              jac.index_put_({Slice(), Slice(1, 2), 0}, one);
              // x2;
              jac.index_put_({Slice(), Slice(2, 3), 3}, one);
              //-1ambda*x1.exp()
              jac.index_put_({Slice(), Slice(3, 4), 2}, -lambda*x1.exp());
              
              return jac;
            }


            /**
             * Dynamics calculated for the augmented Langrangian formulation
             */
            TensorDual dynsbratu(const TensorDual &t,
                                 const TensorDual &y,
                                 const TensorDual &params)
            {
              // auto dynsv = evalDynsDual<double>(y, W, hamiltonian);
              auto x = y.index({Slice(), Slice(0, 2)});
              // std::cerr << "Control = " << ustar << std::endl;
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
              auto lambda = params.index({Slice(), Slice(0,1)});
              //H = p1*x2-p2*lambda*exp(x1)+1

              auto dx1dt = x2;
              auto dx2dt = -lambda*exp(x1);

              auto dyns = TensorDual::cat({dx1dt, dx2dt});

              return dyns;
            }


            TensorMatDual jacbratu(const TensorDual &t,
                                   const TensorDual &y,
                                   const TensorDual &params)
            {
              // auto jacv = evalJacDual<double>(y, W, hamiltonian);
              // return jacv;
              auto x = y.index({Slice(), Slice(0, 2)});
              auto x2 = x.index({Slice(), Slice(1, 2)});
              auto x1 = x.index({Slice(), Slice(0, 1)});
      
              auto lambda = params.index({Slice(), Slice(0,1)});
  
              auto jac = TensorMatDual(torch::zeros({y.r.size(0), 2, 2}, torch::kFloat64),
                                       torch::zeros({y.r.size(0), 2, 2, y.d.size(2)}, torch::kFloat64));
              auto one = TensorDual::ones_like(x1);
              // x2;
              jac.index_put_({Slice(), Slice(0, 1), 1}, one);
              //-1ambda*x1.exp()
              jac.index_put_({Slice(), Slice(1, 2), 0}, -lambda*x1.exp());
              
              return jac;
            }

            /**
             * Radau example using the Bratu
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
                       mint_bratu_propagate(const torch::Tensor &xic,
                                            const torch::Tensor &xf,
                                            const torch::Tensor &x,
                                            const torch::Tensor &lambdap,
                                            const torch::Tensor &mup,           
                                            const torch::Tensor &params)
            {
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
              auto ustartd = calc_control(p10td, p20td, x10td, x20td);
              //p10td = calc_p1(p10td, p20td, x10td, x20td, ustartd);
              //H=p1*x2+p2*ustar*(1-x1*x1)*x2-p2*x1+1;
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

              janus::RadauTeD r(dynsbratu, jacbratu, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
              // Call the solve method of the Radau5 class
              auto rescode = r.solve();
              //Check the return codes
              auto m = rescode != 0;
              if ( m.any().item<bool>())
              {
                //The solver fails gracefully so we just provide a warning message here
                std::cerr << "Solver failed to converge for some samples" << std::endl;
              }
              //Extract the desired final states
              auto x1f = xf.index({Slice(), Slice(0,1)});
              auto x2f = xf.index({Slice(), Slice(1,2)});
              
              
              //Record the projected final values
              auto p1pf = r.y.index({Slice(), Slice(0, 1)});
              auto p2pf = r.y.index({Slice(), Slice(1, 2)});
              auto x1pf = r.y.index({Slice(), Slice(2, 3)});
              auto x2pf = r.y.index({Slice(), Slice(3, 4)});
              //auto ustarpf = calc_control(p1pf, p2pf, x1pf, x2pf);
              auto p0 = y0.r.index({Slice(), Slice(0, 2)});
              auto c1x = x1pf-x1f;
              auto c2x = x2pf-x2f;
              //auto [u1starf, u2starf] = calc_control(p1pf, p2pf, x1pf, x2pf);

              

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(0,1)}),c1x)
                       -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(1,2)}),c2x)
                       +(one+ 0.5*TensorDual::einsum("mi, mi->mi",mup,c1x.square())
                       + 0.5*TensorDual::einsum("mi, mi->mi",mup,c2x.square())).log();


              auto grads = torch::zeros_like(x);
              grads.index_put_({Slice(), 0}, f.d.index({Slice(), 0, 0})); // p1
              grads.index_put_({Slice(), 1}, f.d.index({Slice(), 0, 1})); // p2
              grads.index_put_({Slice(), 2}, f.d.index({Slice(), 0, 4})); // ft
              
              
              auto errors = torch::cat({c1x.r, c2x.r}, 1);
              auto error_norm = torch::cat({c1x.r, c2x.r}, 1).norm();
              //The jacobian is block diagonal
              auto jac = torch::zeros({M, 2, 3}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, c1x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, c1x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(0, 1), Slice(2, 3)}, c1x.d.index({Slice(), Slice(0,1), Slice(4, 5)})); // ft

              jac.index_put_({Slice(), Slice(1, 2), Slice(0, 1)}, c2x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(1, 2), Slice(1, 2)}, c2x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(1, 2), Slice(2, 3)}, c2x.d.index({Slice(), Slice(0,1), Slice(4, 5)})); // ft
           
              return std::make_tuple(f.r, grads, errors, error_norm, jac);
            }


            /**
             * Radau example using the Bratu
             * with sensitivity calculations utilizing dual numbers
             * to calculate the gradients of the augmented Langrangian function
             * The function returns the residuals of the expected
             * end state wrt x1f x2f
             * using initial conditions and tf as the input variables (x)
             * The relationship is defined by the necessary conditions
             * of optimality as defined by the Variational approach to
             * optimal control
             */
            std::tuple<torch::Tensor, torch::Tensor, 
                       torch::Tensor, torch::Tensor,
                       torch::Tensor> 
                       bratu_propagate(const torch::Tensor &xic,
                                       const torch::Tensor &xf,
                                       const torch::Tensor &x,
                                       const torch::Tensor &lambda,
                                       const torch::Tensor &lambdap,
                                       const torch::Tensor &mup,           
                                       const torch::Tensor &params)
            {
              // set the device
              // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
              int M = x.size(0);
              int N = 2; // Length of the state space vector in order [x1, x2]
              int D = 3; // Length of the dual vector in order [x1, x2, tf]
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
              auto x10td = TensorDual(xic.index({Slice(), Slice(0, 1)}), torch::zeros({M, 1, D}, x.options()));
              x10td.d.index_put_({Slice(), 0, 0}, 1.0); // This is an independent variable whose sensitivity we are interested in
              auto x20td = TensorDual(xic.index({Slice(), Slice(1, 2)}), torch::zeros({M, 1, D}, x.options()));
              x20td.d.index_put_({Slice(), 0, 1}, 1.0); // This is an independent variable whose sensitivity we are interested in
              //p10td = calc_p1(p10td, p20td, x10td, x20td, ustartd);
              //H=p1*x2+p2*ustar*(1-x1*x1)*x2-p2*x1+1;
              auto ft = TensorDual(x.index({Slice(), Slice(2, 3)}), torch::zeros({M, 1, D}, x.options()));
              ft.d.index_put_({Slice(), 0, 2}, 1.0); // Set the dependency to itself

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
              auto paramsc = TensorDual::ones_like(x10td)*lambda;

              janus::RadauTeD r(dynsbratu, jacbratu, tspanc, yc, options, paramsc); // Pass the correct arguments to the constructor
              // Call the solve method of the Radau5 class
              auto rescode = r.solve();
              //Check the return codes
              auto m = rescode != 0;
              if ( m.any().item<bool>())
              {
                //The solver fails gracefully so we just provide a warning message here
                std::cerr << "Solver failed to converge for some samples" << std::endl;
              }
              //Extract the desired final states
              auto x1f = xf.index({Slice(), Slice(0,1)});
              auto x2f = xf.index({Slice(), Slice(1,2)});
              
              
              //Record the projected final values
              auto x1pf = r.y.index({Slice(), Slice(0, 1)});
              auto x2pf = r.y.index({Slice(), Slice(1, 2)});
              auto c1x = x1pf-x1f;
              auto c2x = x2pf-x2f;
              //auto [u1starf, u2starf] = calc_control(p1pf, p2pf, x1pf, x2pf);

              

              // The Hamiltonian is zero at the terminal time
              // in principle but this may not be the always the case
              // Now add the augmented Lagrangian term
              auto f = -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(0,1)}),c1x)
                       -TensorDual::einsum("mi, mi->mi",lambdap.index({Slice(), Slice(1,2)}),c2x)
                       +(one+ 0.5*TensorDual::einsum("mi, mi->mi",mup,c1x.square())
                       + 0.5*TensorDual::einsum("mi, mi->mi",mup,c2x.square())).log();


              auto grads = torch::zeros_like(x);
              grads.index_put_({Slice(), 0}, f.d.index({Slice(), 0, 0})); // p1
              grads.index_put_({Slice(), 1}, f.d.index({Slice(), 0, 1})); // p2
              grads.index_put_({Slice(), 2}, f.d.index({Slice(), 0, 2})); // The independent variable
              
              
              auto errors = torch::cat({c1x.r, c2x.r}, 1);
              auto error_norm = torch::cat({c1x.r, c2x.r}, 1).norm();
              //The jacobian is block diagonal
              auto jac = torch::zeros({M, 2, 3}, x.options());
              jac.index_put_({Slice(), Slice(0, 1), Slice(0, 1)}, c1x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(0, 1), Slice(1, 2)}, c1x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(0, 1), Slice(2, 3)}, c1x.d.index({Slice(), Slice(0,1), Slice(2, 3)})); // ft

              jac.index_put_({Slice(), Slice(1, 2), Slice(0, 1)}, c2x.d.index({Slice(), Slice(0,1), Slice(0, 1)})); // p1
              jac.index_put_({Slice(), Slice(1, 2), Slice(1, 2)}, c2x.d.index({Slice(), Slice(0,1), Slice(1, 2)})); // p2
              jac.index_put_({Slice(), Slice(1, 2), Slice(2, 3)}, c2x.d.index({Slice(), Slice(0,1), Slice(2, 3)})); // ft
           
              return std::make_tuple(f.r, grads, errors, error_norm, jac);
            }



      }
    }
  }

} // namespace janus
#endif