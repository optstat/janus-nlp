/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#include <janus/radauted.hpp>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>
#include "../../src/cpp/lnsrchted.hpp"
#include "../../src/cpp/newtted.hpp"
#include "../../src/cpp/newtte.hpp"
#include "../../src/cpp/lnsrchte.hpp"
#include "matplotlibcpp.h"

using namespace janus;
namespace plt = matplotlibcpp;


/**
 * Radau example using the Van der Pol oscillator 
 * Using the Hamiltonian with dual number approach to calcuate the dynamics and
 * the Jacobian
*/
using Slice = torch::indexing::Slice;
//Global parameters for simplicity
double W = 1.0;
double x1f = 1.0;
double x2f = -1.25;
double x10 = 2.0;
double x20 = 0.0;
double ft = 1.0;

//Guesses for the initial values of the Lagrange multipliers
double p10 = -10.0;
double p20 = 10.0;



 

TensorDual control_function_dual(const TensorDual& u,
                                 const TensorDual& params) 
{
  auto x1 = params.index({Slice(), Slice(0,1)});
  auto x2 = params.index({Slice(), Slice(1,2)});
  auto p1 = params.index({Slice(), Slice(2,3)});
  auto p2 = params.index({Slice(), Slice(3,4)});
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





TensorDual control_dual(const TensorDual& u_guess,
                        const TensorDual& x1, 
                        const TensorDual& x2,
                        const TensorDual& p1,
                        const TensorDual& p2) 
{
  //We have to solve
  
  TensorDual params =TensorDual::cat({p1, p2, x1, x2});
  TensorDual umin = TensorDual::zeros_like(x1);
  TensorDual umax = TensorDual::zeros_like(x1);  
  umin.index_put_({Slice(), Slice(0,1)}, 0.001);
  umax.index_put_({Slice(), Slice(0,1)}, 100.0);
  TensorDual u = u_guess.clone();
  

  auto res= newtTeD(u, params, umin, umax, control_function_dual, control_grad_function_dual);
  return std::get<0>(res);
}








TensorDual hamiltonian_dual(const TensorDual& x, 
                            const TensorDual& p, 
                            double W) {
  TensorDual p1 = p.index({Slice(), 0});  
  TensorDual p2 = p.index({Slice(), 1});  
  TensorDual x1 = x.index({Slice(), 0});  
  TensorDual x2 = x.index({Slice(), 1});
  TensorDual u_guess = TensorDual::ones_like(x1);
  auto u = control_dual(u_guess, x1, x2, p1, p2);  
  auto H = p1*x2+p2*u*(1-x1*x1)*x2+W*(u.log().square()/2+u*u/2)+1; //Return the through copy elision
  return H; //Return the through copy elision
}


torch::Tensor hamiltonian(const torch::Tensor& x, 
                          const torch::Tensor& p, 
                          double W) {
  //Wrap the call to the dual number version
  int M = x.size(0);
  int N = x.size(1);
  torch::Tensor dummy = torch::zeros({M, N, 1}, torch::kFloat64);
  auto x_dual = TensorDual(x.clone(), dummy.clone());
  auto p_dual = TensorDual(p.clone(), dummy.clone());
  auto H_dual = hamiltonian_dual(x_dual, p_dual, W);
  return H_dual.r;
  
}



/**
 * Dynamics calculated according the hamiltonian method
 */
TensorDual vdpdyns_ham(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  auto dyns=  <double>(y, W, hamiltonian);
  //std::cerr << "dyns=";
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



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> vdpEvents(torch::Tensor& t, 
                                                                  torch::Tensor& y, 
                                                                  torch::Tensor& params) {
    int M = y.size(0);
    //return empty tensors
    torch::Tensor E = y.index({Slice(), 0});
    torch::Tensor Stop = torch::tensor({M, false}, torch::TensorOptions().dtype(torch::kBool));
    auto mask = (y.index({Slice(), 1}) == 0.0);
    Stop.index_put_({mask}, true);
    torch::Tensor Slope = torch::tensor({M, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    return std::make_tuple(t, E, Stop, Slope);
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
  int M = x.size(0); //Number of samples
  int N = 4; //Number of variables
  int D = 1; //Length of the dual vector keep it small to speed up the calculation since we don't need 
             //dual numbers here but to avoid writing two different methods for dynamics and jacobian
  auto device = x.device();

  TensorDual y = TensorDual(torch::zeros({M, N}, torch::kF64).to(device), 
                            torch::zeros({M,N,D}, torch::kF64).to(device));
  y.r.index_put_({Slice(), Slice(0,1)}, x.index({Slice(), Slice(0,1)}));  //p1
  y.r.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), Slice(1,2)}));  //p2
  y.r.index_put_({Slice(), Slice(2,3)}, x10);                       //x1
  y.r.index_put_({Slice(), Slice(3,4)}, x20);                       //x2

 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), 
                                torch::zeros({M,2,1}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), Slice{0,1}}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), Slice(2,3)}));
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  //Give the ODE integrator modest tolerances and a maximum number of steps
  //to ensure that reasonable execution times
  options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
  options.MaxNbrStep = 200; 

  //Create l values this is a C++ requirement for non constant references
  auto tspanc = tspan.clone();
  auto yc = y.clone();
  auto paramsc = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));

  janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, yc, options, paramsc);   // Pass the correct arguments to the constructor`
  int rescode = r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  
  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return a large result to make sure the solver does not fail
    return torch::ones_like(x)*1.0e6;
  }
  auto pf = r.y.index({Slice(), Slice(0,2)});
  auto xf = r.y.index({Slice(), Slice(2,4)});
  //Now calculate the boundary conditionsx
  auto x1delta = r.y.index({Slice(), Slice(2,3)})-x1f;
  auto x2delta = r.y.index({Slice(), Slice(3,4)})-x2f;
  auto Hf = hamiltonian_dual(xf, pf, W);
  torch::Tensor res = x*0.0;
  //The hamiltonian is zero at the terminal time 
  //because this is a minimum time problem
  res.index_put_({Slice(), Slice(0,1)}, x1delta.r);
  res.index_put_({Slice(), Slice(1,2)}, x2delta.r);
  res.index_put_({Slice(), Slice(2,3)}, Hf.r);
  std::cerr << "propagation result=";
  janus::print_tensor(res);

  return res;

}


/**
 * Calculate the Jacobian of the propagation function
 * 
 */
torch::Tensor jac_eval(const torch::Tensor& x, const torch::Tensor& params) {
  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int M = x.size(0);
  int N = 4;//Length of the state space vector in order [p1, p2, x1, x2] 
  int D = 5;//Length of the dual vector in order [p1, p2, x1, x2, tf]


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, N}, x.options()), torch::zeros({M,N,D},x.options()));
  y.r.index_put_({Slice(), Slice(0,1)}, x.index({Slice(), Slice(0,1)}));
  y.r.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), Slice(1,2)}));
  y.r.index_put_({Slice(), Slice(2,3)}, x10);
  y.r.index_put_({Slice(), Slice(3,4)}, x20);

  //Set the sensitivities to the initial conditions
  y.d.index_put_({Slice(), 0, 0}, 1.0);
  y.d.index_put_({Slice(), 1, 1}, 1.0);
  y.d.index_put_({Slice(), 2, 2}, 1.0);
  y.d.index_put_({Slice(), 3, 3}, 1.0);
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, x.options()), torch::zeros({M,2,N}, x.options()));
  tspan.r.index_put_({Slice(), Slice(0,1)}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), Slice(2,3)}));
  tspan.d.index_put_({Slice(), 1, -1}, 1.0);  //Sensitivity to the final time
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  options.RelTol = torch::tensor({1e-3}, x.options());
  options.AbsTol = torch::tensor({1e-6}, x.options());
  
  options.MaxNbrStep = 200;
  auto tspanc = tspan.clone();
  auto yc = y.clone();
  auto paramsc = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));


  janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, yc, options, paramsc);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  int rescode = r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  torch::Tensor jacVal = torch::zeros({M, 3, 3}, torch::kFloat64); 

  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return default zeros.  These values will not be used
    return jacVal;
  }

  //Extract the final values from the solver
  auto pf = r.y.index({Slice(), Slice(0,2)});
  auto xf = r.y.index({Slice(), Slice(2,4)});
  std::cerr << "xf=";
  janus::print_dual(xf);
  auto x1delta = r.y.index({Slice(), Slice(2,3)})-x1f;
  std::cerr << "x1delta=";
  janus::print_dual(x1delta);
  auto x2delta = r.y.index({Slice(), Slice(3,4)})-x2f;
  std::cerr << "x2delta=";
  janus::print_dual(x2delta);
  auto Hf = hamiltonian_dual(xf, pf, W);
  jacVal.index_put_({Slice(), 0, 0}, x1delta.d.index({Slice(),0,0}));
  jacVal.index_put_({Slice(), 0, 1}, x1delta.d.index({Slice(),0,1}));
  jacVal.index_put_({Slice(), 0, 2}, x1delta.d.index({Slice(),0,N-1}));
  std::cerr << "jacobian result=";
  janus::print_tensor(jacVal);


  return jacVal;

}


/**
 * The goal of this example is to solve for a minimum time optimal control problem
 * that can become very stiff very suddenly requiring the use of novel dual number
 * sensitivity and global optimization techniques to solve the problem
 */

int main(int argc, char *argv[])
{
  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  int M =1;  //Number of samples
  int N =2;  //Number of costates
  int D = N+1; //Dimension of the dual numbers [p1, p2, tf]
  //The problem has three unknowns p1 p2 and final time tf
  /*
  -28.8603   1.4887   6.1901*/

  auto device = torch::kCPU;

  torch::Tensor y0 = torch::ones({1, D}, torch::dtype(torch::kFloat64)).to(device);
  for ( int i=0; i < M; i++) {
    y0.index_put_({i, 0}, p10+0.1*i);  //p1
    y0.index_put_({i, 1}, p20+0.1*i);  //p2
    y0.index_put_({i, 2}, ft);         //ft
  }

  
  torch::Tensor params = torch::zeros_like(y0);
  //Impose limits on the state space
  torch::Tensor ymin = torch::zeros_like(y0);
  ymin.index_put_({Slice(), 0}, -10000.0);
  ymin.index_put_({Slice(), 1}, -10000.0);
  ymin.index_put_({Slice(), 2}, 0.001);
  torch::Tensor ymax = torch::zeros_like(y0);
  ymax.index_put_({Slice(), 0}, 10000.0);
  ymax.index_put_({Slice(), 1}, 10000.0);
  ymax.index_put_({Slice(), 2}, 100.0);

  
  auto res   = newtTe(y0, params, ymin, ymax, propagate, jac_eval);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  std::cerr << "roots=" << roots << "\n";
  std::cerr << "check=" << check << "\n";
  auto errors = Jfunc(propagate(roots, params));
  std::cerr << "errors=" << errors << "\n";

  //Now propagate the final solution to get the final sensitivities


  return 0;
}




