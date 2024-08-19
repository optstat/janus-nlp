#ifndef NEWT_DUBINS_EXAMPLE_HPP
#define NEWT_DUBINS_EXAMPLE_HPP
/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#include <torch/torch.h>
#include <janus/radauted.hpp>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>
#include "../../src/cpp/lnsrchte.hpp"
#include "../../src/cpp/newtte.hpp"


using namespace janus;
namespace janus {
  namespace nlp {
    namespace examples {
      namespace dubins {
int M = 2;
/**
 * Radau example using the Van der Pol oscillator 
 * Using the Hamiltonian with dual number approach to calcuate the dynamics and
 * the Jacobian
*/
using Slice = torch::indexing::Slice;
double pi = M_PI;
double W  = 0.01; //Regularization weight
double x1f = 0.0; //Final value of x1
double x2f = 0.0; //Final value of x2
auto x10 = torch::linspace(0.5, 0.6, M, torch::kFloat64); //Initial value of x1
double x20 = 0.0; //Initial value of x2
double x30 = pi;  //Initial value of x3
//Guesses for the initial values of the Lagrange multipliers
//1.1200 -0.0702 -0.0351  0.5252
double p10 = 0.0;
double p20 = 0.0;
double p30 = 0.0;
double ft0 = 0.1;

 





TensorDual hamiltonian_dual(const TensorDual& x, 
                            const TensorDual& p, 
                            double W) {
  TensorDual p1 = p.index({Slice(), 0});  
  TensorDual p2 = p.index({Slice(), 1});
  TensorDual p3 = p.index({Slice(), 2});  
  TensorDual x1 = x.index({Slice(), 0});  
  TensorDual x2 = x.index({Slice(), 1});
  TensorDual x3 = x.index({Slice(), 2});  
  TensorDual one = TensorDual(torch::ones_like(p1.r), torch::zeros_like(p1.d));
  auto H = p1*x3.cos()+p2*x3.sin()-p3*p3/(2*W)+one; //Return the through copy elision
  return H; //Return the through copy elision
}


torch::Tensor hamiltonian(const torch::Tensor& x, 
                          const torch::Tensor& p, 
                   double W) {
  torch::Tensor p1 = p.index({Slice(), 0});  
  torch::Tensor p2 = p.index({Slice(), 1});
  torch::Tensor p3 = p.index({Slice(), 2});  
  torch::Tensor x1 = x.index({Slice(), 0});  
  torch::Tensor x2 = x.index({Slice(), 1});
  torch::Tensor x3 = x.index({Slice(), 2});  
  auto H = p1*x3.cos()+p2*x3.sin()-p3*p3/(2*W)+1; //Return the through copy elision
  return H; //Return the through copy elision
}



/**
 * Dynamics calculated according the hamiltonian method
 */
TensorDual vdpdyns_ham(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  auto dyns= evalDynsDual<double>(y, W, hamiltonian);
  //std::cerr << "dyns=";
  //janus::print_dual(dyns);
  return dyns;
}

/**
 * Dynamics calculated according the hamiltonian method
 */
TensorDual dubinsdyns(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  auto dyns= y*0.0;
  auto p1 = y.index({Slice(), Slice(0,1)});
  auto p2 = y.index({Slice(), Slice(1,2)});
  auto p3 = y.index({Slice(), Slice(2,3)});
  auto x1 = y.index({Slice(), Slice(3,4)});
  auto x2 = y.index({Slice(), Slice(4,5)});
  auto x3 = y.index({Slice(), Slice(5,6)});
  auto zero = p1*0.0;

  dyns.index_put_({Slice(), Slice(0,1)}, zero);
  dyns.index_put_({Slice(), Slice(1,2)}, zero);
  dyns.index_put_({Slice(), Slice(2,3)}, -p1*x3.sin()+p2*x3.cos());
  dyns.index_put_({Slice(), Slice(3,4)}, x3.cos());
  dyns.index_put_({Slice(), Slice(4,5)}, x3.sin());
  dyns.index_put_({Slice(), Slice(5,6)}, -p3/(W));
  //std::cerr << "dyns="
  //janus::print_dual(dyns);
  return dyns;
}


TensorMatDual jac_ham(const TensorDual& t, 
                   const TensorDual& y, 
                   const TensorDual& params) {
  auto jac = evalJacDual<double>(y, W, hamiltonian);
  //std::cerr << "jac_ham=";
  //janus::print_dual(jac);
  return jac;
}

TensorMatDual dubinsjac(const TensorDual& t, 
                   const TensorDual& y, 
                   const TensorDual& params) {
  auto jac = TensorMatDual(torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1)}, torch::kFloat64), 
                           torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1), y.d.size(2)}, torch::kFloat64));
  auto p1 = y.index({Slice(), Slice(0,1)});
  auto p2 = y.index({Slice(), Slice(1,2)});
  auto p3 = y.index({Slice(), Slice(2,3)});
  auto x1 = y.index({Slice(), Slice(3,4)});
  auto x2 = y.index({Slice(), Slice(4,5)});
  auto x3 = y.index({Slice(), Slice(5,6)});
  TensorDual one = TensorDual(torch::ones_like(p1.r), torch::zeros_like(p1.d));
  TensorDual zero = TensorDual(torch::zeros_like(p1.r), torch::zeros_like(p1.d));

  jac.index_put_({Slice(), Slice(2,3), 0}, -x3.sin());
  jac.index_put_({Slice(), Slice(2,3), 1}, x3.cos());

  jac.index_put_({Slice(), Slice(3,4), 5}, -x3.sin()); 
  
  jac.index_put_({Slice(), Slice(4,5), 5}, x3.cos());
  
  jac.index_put_({Slice(), Slice(5,6), 2}, -one/(W));

  return jac;
}




/**
 * Radau example using the Van der Pol oscillator
 * using dual numbers for sensitivities
 * The function returns the residuals of the expected
 * end state wrt x1f x2f and final Hamiltonian value
 * using p10 p20 and tf as the input variables
 * The relationship is defined by the necessary conditions
 * of optimality as defined by the Variational approach to 
 * optimal control
 */
torch::Tensor propagate(const torch::Tensor& x, const torch::Tensor& params) 
{

  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int D = 6; //Number of variables
  int N = 1; //Length of the dual vector.  To keep things simple set it to 1 since we don't need sensitivities
  auto device = x.device();
  auto p10 = x.index({Slice(), 0});
  auto p20 = x.index({Slice(), 1});
  auto p30 = x.index({Slice(), 2});
  int M = x.size(0);

  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::zeros({M,D,N}, torch::kF64).to(device));
  y.r.index_put_({Slice(), 0}, p10);
  y.r.index_put_({Slice(), 1}, p20);
  y.r.index_put_({Slice(), 2}, p30);
  y.r.index_put_({Slice(), 3}, x10.index({Slice(0,M)}));
  y.r.index_put_({Slice(), 4}, x20);
  y.r.index_put_({Slice(), 5}, x30);
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, x.index({Slice(), 3}));
  tspan.d.index_put_({Slice(), 1, N-1}, 1.0); //Sensitivity to the final times
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
  options.MaxNbrStep = 200;


  auto params_dual = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));
  //Create an instance of the Radau5 class
  //Call the solve method of the Radau5 class
  janus::RadauTeD r(dubinsdyns, dubinsjac, tspan, y, options, params_dual);   // Pass the correct arguments to the constructor`
  std::cerr << "Caling ODE solver" << std::endl;
  int rescode = r.solve();
  std::cerr << "ODE solver finished with count" <<  r.count << std::endl;
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  
  auto pf = r.y.index({Slice(), Slice(0,3)});
  auto xf = r.y.index({Slice(), Slice(3,6)});
  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return a large result to make sure the solver does not fail
    return torch::ones({M, 3}, torch::kFloat64)*1.0e6;
  }
  auto x1delta = r.y.index({Slice(), Slice(3,4)})-x1f;
  auto x2delta = r.y.index({Slice(), Slice(4,5)})-x2f;
  auto p3delta = r.y.index({Slice(), Slice(2,3)});
  auto Hf = hamiltonian(xf.r, pf.r, W);
  torch::Tensor res = torch::zeros({M, 4}, torch::kFloat64);
  //The hamiltonian is zero at the terminal time 
  //because this is a minimum time problem
  //Here we have four equations in four unknowns
  res.index_put_({Slice(), 0}, x1delta.r.squeeze(1));
  res.index_put_({Slice(), 1}, x2delta.r.squeeze(1));
  res.index_put_({Slice(), 2}, p3delta.r.squeeze(1));
  res.index_put_({Slice(), 3}, Hf);
  std::cerr << "propagation result=";
  janus::print_tensor(res);

  return res;

}

torch::Tensor jac_eval(const torch::Tensor& x, const torch::Tensor& params) {
  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int D = 6; //Number of variables
  int N = 7; //Length of the dual vector in order [p1, p2, p3, x1, x2, x3, tf]
  auto device = x.device();
  auto p10 = x.index({Slice(), Slice(0,1)});
  auto p20 = x.index({Slice(), Slice(1,2)});
  auto p30 = x.index({Slice(), Slice(2,3)});
  int M = x.size(0);

  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::zeros({M,D,N}, torch::kF64).to(device));
  y.r.index_put_({Slice(), Slice(0,1)}, p10);
  y.r.index_put_({Slice(), Slice(1,2)}, p20);
  y.r.index_put_({Slice(), Slice(2,3)}, p30);
  y.r.index_put_({Slice(), Slice(3,4)}, x10.index({Slice(0,M)}).unsqueeze(1));
  y.r.index_put_({Slice(), Slice(4,5)}, x20);
  y.r.index_put_({Slice(), Slice(5,6)}, x30);
  y.d.index_put_({Slice(), 0, 0}, 1.0);
  y.d.index_put_({Slice(), 1, 1}, 1.0);
  y.d.index_put_({Slice(), 2, 2}, 1.0);
  y.d.index_put_({Slice(), 3, 3}, 1.0);
  y.d.index_put_({Slice(), 4, 4}, 1.0);
  y.d.index_put_({Slice(), 5, 5}, 1.0);

 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), Slice(3,4)}));
  tspan.d.index_put_({Slice(), 1, N-1}, 1.0); //Sensitivity to the final times
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
  options.MaxNbrStep = 1000;


  auto params_dual = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));
  //Create an instance of the Radau5 class
  //Call the solve method of the Radau5 class
  janus::RadauTeD r(dubinsdyns, dubinsjac, tspan, y, options, params_dual);   // Pass the correct arguments to the constructor`
  int rescode = r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  
  auto pf = r.y.index({Slice(), Slice(0,3)});
  auto xf = r.y.index({Slice(), Slice(3,6)});
  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return a large result to make sure the solver does not fail
    return torch::ones({M, 3}, torch::kFloat64)*1.0e6;
  }
  auto x1delta = r.y.index({Slice(), Slice(3,4)})-x1f;
  auto x2delta = r.y.index({Slice(), Slice(4,5)})-x2f;
  auto p3delta = r.y.index({Slice(), Slice(2,3)});
  auto Hf = hamiltonian_dual(xf, pf, W);
  torch::Tensor res = torch::zeros({M, 4}, torch::kFloat64);
  //The hamiltonian is zero at the terminal time 
  //because this is a minimum time problem
  //Here we have four equations in four unknowns
  res.index_put_({Slice(), Slice(0,1)}, x1delta.r);
  res.index_put_({Slice(), Slice(1,2)}, x2delta.r);
  res.index_put_({Slice(), Slice(2,3)}, p3delta.r);
  res.index_put_({Slice(), Slice(3,4)}, Hf.r);
  std::cerr << "propagation result=";
  janus::print_tensor(res);

  //Calculate the jacobian
  auto jacVal = torch::zeros({M, 4, 4}, torch::kFloat64);


  jacVal.index_put_({Slice(), 0, 0}, x1delta.d.index({Slice(), 0, 0}));
  jacVal.index_put_({Slice(), 0, 1}, x1delta.d.index({Slice(), 0, 1}));
  jacVal.index_put_({Slice(), 0, 2}, x1delta.d.index({Slice(), 0, 2}));
  jacVal.index_put_({Slice(), 0, 3}, x1delta.d.index({Slice(), 0, -1}));
  
  jacVal.index_put_({Slice(), 1, 0}, x2delta.d.index({Slice(), 0, 0}));
  jacVal.index_put_({Slice(), 1, 1}, x2delta.d.index({Slice(), 0, 1}));
  jacVal.index_put_({Slice(), 1, 2}, x2delta.d.index({Slice(), 0, 2}));
  jacVal.index_put_({Slice(), 1, 3}, x2delta.d.index({Slice(), 0, -1}));
  
  jacVal.index_put_({Slice(), 2, 0}, p3delta.d.index({Slice(), 0, 0}));
  jacVal.index_put_({Slice(), 2, 1}, p3delta.d.index({Slice(), 0, 1}));
  jacVal.index_put_({Slice(), 2, 2}, p3delta.d.index({Slice(), 0, 2}));
  jacVal.index_put_({Slice(), 2, 3}, p3delta.d.index({Slice(), 0, -1}));
  
  
  jacVal.index_put_({Slice(), 3, 0}, Hf.d.index({Slice(), 0, 0}));
  jacVal.index_put_({Slice(), 3, 1}, Hf.d.index({Slice(), 0, 1}));
  jacVal.index_put_({Slice(), 3, 2}, Hf.d.index({Slice(), 0, 2}));
  jacVal.index_put_({Slice(), 3, 3}, Hf.d.index({Slice(), 0, -1}));
  std::cerr << "jacobian result=";
  janus::print_tensor(jacVal);


  return jacVal;

}


//Create a main method for testing

int solve()
{
  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  int D =6;
  /*
  -28.8603   1.4887   6.1901*/

  torch::Tensor x0 = torch::ones({M, 4}, torch::dtype(torch::kFloat64));
  for ( int i=0; i < M; i++) {
    //Adjust the guesses accordingly
    x0.index_put_({i, 0}, p10);  //p1
    x0.index_put_({i, 1}, p20);  //p2
    x0.index_put_({i, 2}, p30);  //p3
    x0.index_put_({i, 3}, ft0);  //ft
  }
  torch::Tensor params = torch::zeros_like(x0);
  //Impose limits on the state space
  torch::Tensor xmin = torch::zeros_like(x0);
  xmin.index_put_({Slice(), 0}, -1000.0);
  xmin.index_put_({Slice(), 1}, -1000.0);
  xmin.index_put_({Slice(), 2}, -1000.0);
  xmin.index_put_({Slice(), 3}, 0.0001);
  torch::Tensor xmax = torch::zeros_like(x0);
  xmax.index_put_({Slice(), 0}, 1000.0);
  xmax.index_put_({Slice(), 1}, 1000.0);
  xmax.index_put_({Slice(), 2}, 1000.0);
  xmax.index_put_({Slice(), 3}, 10.0);

  
  auto res = newtTe(x0, params, xmin, xmax, propagate, jac_eval);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  std::cerr << "roots=" << roots << "\n";
  std::cerr << "check=" << check << "\n";
  auto errors = Jfunc(propagate(roots, params));
  std::cerr << "errors=" << errors << "\n";


  return 0;
}
    }
  }
  }
} // namespace dubins


#endif