#ifndef NEWTED_CONTROL_EXAMPLE_HPP
#define NEWTED_CONTROL_EXAMPLE_HPP
/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control only using the dual number approac
 */
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>
#include "../../src/cpp/lnsrchted.hpp"
#include "../../src/cpp/newtted.hpp"


namespace janus {
namespace nlp {
namespace examples {
  namespace vdpc {

using namespace janus;

int M = 1; 
/**
 * Simple example to illustrate the use of Global Newton method for a simple optimal control
 * that illustrates the use of this method to calculate sensitivities wrt to the initial conditions
*/
using Slice = torch::indexing::Slice;
double W  = 0.01; //Regularization weight
double x10 = 0.1; //Initial value of x1
double x20 = 0.1; //Initial value of x2
//Guesses for the initial values of the Lagrange multipliers
//1.1200 -0.0702 -0.0351  0.5252
double p10 = 0.1;
double p20 = 0.1;
double p30 = 0.1;
double ft0 = 0.1;

 

TensorDual control_function_dual(const TensorDual& u,
                                 const TensorDual& params) 
{
  auto p1 = params.index({Slice(), Slice(0,1)});
  auto p2 = params.index({Slice(), Slice(1,2)});
  auto x1 = params.index({Slice(), Slice(2,3)});
  auto x2 = params.index({Slice(), Slice(3,4)});
  //We have to solve

  auto res =u.log()/u+u+p2*(x2*(1-x1*x1)-x1)*(1/W);

  return res; //Return the through copy elision
}


TensorMatDual control_grad_function_dual(const TensorDual& u,
                                         const TensorDual& params) 
{
  auto x1 = params.index({Slice(), Slice(0,1)});
  auto x2 = params.index({Slice(), Slice(1,2)});
  auto p1 = params.index({Slice(), Slice(2,3)});
  auto p2 = params.index({Slice(), Slice(3,4)});
  //We have to solve
  auto res = -u.log()/(u*u)+(u*u).reciprocal()+1;
  auto jac = res.unsqueeze(2);

  return jac; //Return the through copy elision
} 





/**
 * Newton example using the Van der Pol oscillator controlled Hamiltonian
 * using dual numbers for sensitivities
 * The function returns the optimal control for a single instance of
 * x and p
 */
TensorDual propagate(const TensorDual& u, const TensorDual& params) 
{

    return control_function_dual(u, params);
}

TensorMatDual jac_eval(const TensorDual& u, const TensorDual& params) {
    return control_grad_function_dual(u, params);
}


//Create a main method for testing

int solve()
{
  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  int M=1;
  int N=4;
  int D=5;
  TensorDual u0 = TensorDual(torch::ones({M, 1}, torch::dtype(torch::kFloat64)),
                             torch::zeros({M, 1, D}, torch::dtype(torch::kFloat64)));
  //Set the dual parts to one
  TensorDual params = TensorDual(torch::zeros({M, D}, torch::dtype(torch::kFloat64)),
                                 torch::zeros({M, D, D}, torch::dtype(torch::kFloat64)));
  params.r.index_put_({Slice(), Slice(0,1)}, p10);
  params.r.index_put_({Slice(), Slice(1,2)}, p20);
  params.r.index_put_({Slice(), Slice(2,3)}, x10);
  params.r.index_put_({Slice(), Slice(3,4)}, x20);
  params.d.index_put_({Slice(), Slice(0,1), Slice(0,1)}, 1.0);
  params.d.index_put_({Slice(), Slice(1,2), Slice(1,2)}, 1.0);
  params.d.index_put_({Slice(), Slice(2,3), Slice(2,3)}, 1.0);
  params.d.index_put_({Slice(), Slice(3,4), Slice(3,4)}, 1.0);

  //Set the dual numbers so we can check with finite differences whether the calculation
  //was correct
  auto paramsc = params.clone();

  TensorDual umin = TensorDual(-10*torch::ones({M, 1}, torch::dtype(torch::kFloat64)),
                             torch::zeros({M, 1, D}, torch::dtype(torch::kFloat64)));

  TensorDual umax = TensorDual(10*torch::ones({M, 1}, torch::dtype(torch::kFloat64)),
                             torch::zeros({M, 1, D}, torch::dtype(torch::kFloat64)));

  auto res = newtTeD(u0, params, umin, umax, propagate, jac_eval);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  auto errors = Jfunc(propagate(roots, params));
  std::cerr << "roots=";
  janus::print_dual(roots);
  std::cerr << "errors=" << errors << std::endl;
  //Now check against finite differences
  auto one = torch::ones({M, 1}, torch::dtype(torch::kFloat64));
  auto maxres =torch::max(params.index({Slice(), Slice(0,1)}).r, one);
  auto h = 1.0e-8*maxres;
  auto params1p = params.clone();
  params1p.r.index_put_({Slice(), Slice(0,1)}, params1p.index({Slice(), Slice(0,1)}).r+h); 
  auto res1p = newtTeD(u0, params1p, umin, umax, propagate, jac_eval);
  auto params1m = params.clone();
  params1m.r.index_put_({Slice(), Slice(0,1)}, params1m.index({Slice(), Slice(0,1)}).r-h); 
  auto res1m = newtTeD(u0, params1m, umin, umax, propagate, jac_eval);
  auto grad1 = (std::get<0>(res1p).r-std::get<0>(res1m).r)/(2*h);
  std::cerr << "grad1=" << grad1 << std::endl;
  std::cerr << "check1=" << roots.d.index({Slice(), Slice(0,1), Slice(0,1)}) << std::endl;
  assert(torch::allclose(grad1, roots.d.index({Slice(), Slice(0,1), Slice(0,1)}), 1.0e-6, 1.0e-6));

  maxres =torch::max(params.index({Slice(), Slice(1,2)}).r, one);
  h = 1.0e-8*maxres;
  auto params2p = params.clone();
  params2p.r.index_put_({Slice(), Slice(1,2)}, params2p.index({Slice(), Slice(1,2)}).r+h); 
  auto res2p = newtTeD(u0, params2p, umin, umax, propagate, jac_eval);
  auto params2m = params.clone();
  params2m.r.index_put_({Slice(), Slice(1,2)}, params1m.index({Slice(), Slice(1,2)}).r-h); 
  auto res2m = newtTeD(u0, params2m, umin, umax, propagate, jac_eval);
  auto grad2 = (std::get<0>(res2p).r-std::get<0>(res2m).r)/(2*h);
  std::cerr << "grad2=" << grad2 << std::endl;
  std::cerr << "check2=" << roots.d.index({Slice(), Slice(0,1), Slice(1,2)}) << std::endl;
  assert(torch::allclose(grad2, roots.d.index({Slice(), Slice(0,1), Slice(1,2)}), 1.0e-6, 1.0e-6));

  maxres =torch::max(params.index({Slice(), Slice(2,3)}).r, one);
  h = 1.0e-8*maxres;
  auto params3p = params.clone();
  params3p.r.index_put_({Slice(), Slice(2,3)}, params2p.index({Slice(), Slice(2,3)}).r+h); 
  auto res3p = newtTeD(u0, params3p, umin, umax, propagate, jac_eval);
  auto params3m = params.clone();
  params3m.r.index_put_({Slice(), Slice(2,3)}, params1m.index({Slice(), Slice(2,3)}).r-h); 
  auto res3m = newtTeD(u0, params3m, umin, umax, propagate, jac_eval);
  auto grad3 = (std::get<0>(res3p).r-std::get<0>(res3m).r)/(2*h);
  std::cerr << "grad3=" << grad3 << std::endl;
  std::cerr << "check3=" << roots.d.index({Slice(), Slice(0,1), Slice(2,3)}) << std::endl;
  assert(torch::allclose(grad3, roots.d.index({Slice(), Slice(0,1), Slice(2,3)}), 1.0e-6, 1.0e-6));

  maxres =torch::max(params.index({Slice(), Slice(3,4)}).r, one);
  h = 1.0e-8*maxres;
  auto params4p = params.clone();
  params4p.r.index_put_({Slice(), Slice(3,4)}, params2p.index({Slice(), Slice(3,4)}).r+h); 
  auto res4p = newtTeD(u0, params4p, umin, umax, propagate, jac_eval);
  auto params4m = params.clone();
  params4m.r.index_put_({Slice(), Slice(3,4)}, params1m.index({Slice(), Slice(3,4)}).r-h); 
  auto res4m = newtTeD(u0, params4m, umin, umax, propagate, jac_eval);
  auto grad4 = (std::get<0>(res4p).r-std::get<0>(res4m).r)/(2*h);
  std::cerr << "grad4=" << grad4 << std::endl;
  std::cerr << "check4=" << roots.d.index({Slice(), Slice(0,1), Slice(3,4)}) << std::endl;
  assert(torch::allclose(grad4, roots.d.index({Slice(), Slice(0,1), Slice(3,4)}), 1.0e-6, 1.0e-6));

  return 0;
}
}
}
}
} // namespace vdp


#endif