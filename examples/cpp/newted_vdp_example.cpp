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
#include "matplotlibcpp.h"

using namespace janus;
namespace plt = matplotlibcpp;


/**
 * Radau example using the Van der Pol oscillator 
 * Using the Hamiltonian with dual number approach to calcuate the dynamics and
 * the Jacobian
*/
using Slice = torch::indexing::Slice;
double W = 1.0;
double x1f = 1.0;
double x2f = -1.25;
double x10 = 2.0;
double x20 = 0.0;
double ft = 1.0;
double mu=1.0;
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


TensorDual control_grad_function_dual(const TensorDual& u,
                                         const TensorDual& params) 
{
  auto x1 = params.index({Slice(), 0});
  auto x2 = params.index({Slice(), 1});
  auto p1 = params.index({Slice(), 2});
  auto p2 = params.index({Slice(), 3});
  //We have to solve
  auto res = -u.log()/(u*u)+(u*u).reciprocal()+1;

  return res; //Return the through copy elision
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
  umax.index_put_({Slice(), Slice(0,1)}, 1000.0);
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
  torch::Tensor dummy = torch::zeros({M, 1}, torch::kFloat64);
  auto x_dual = TensorDual(x.clone(), dummy.clone());
  auto p_dual = TensorDual(p.clone(), dummy.clone());
  auto H_dual = hamiltonian_dual(x_dual, p_dual, W);
  return H_dual.r;
  
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
TensorDual propagate(const TensorDual& x, const TensorDual& params) 
{

  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int M = x.r.size(0); //Number of samples
  int D = 3; //Number of variables
  int N = 5; //Length of the dual vector in order [p1, p2, x1, x2, tf]
  auto device = x.device();

  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, 4}, torch::kF64).to(device), torch::zeros({M,4,N}, torch::kF64).to(device));
  for (int i=0; i < M; i++) {
    y.index_put_({i, 0}, x.index({i, 0}));
    y.index_put_({i, 1}, x.index({i, 1}));
    y.index_put_({i, 2}, x10);
    y.index_put_({i, 3}, x20);
    y.index_put_({i, 0, 0}, 1.0);
    y.index_put_({i, 1, 1}, 1.0);
    y.index_put_({i, 2, 2}, 1.0);
    y.index_put_({i, 3, 3}, 1.0);
  }
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), Slice(1,2)}, x.index({Slice(), 2}));
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
  options.MaxNbrStep = 1000;


  //Create an instance of the Radau5 class
  //Call the solve method of the Radau5 class
  auto tspanc = tspan.clone();
  auto yc = y.clone();
  auto paramsc = params.clone();
  janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, yc, options, paramsc);   // Pass the correct arguments to the constructor`
  int rescode = r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  
  auto pf = r.y.index({Slice(), Slice(0,2)});
  auto xf = r.y.index({Slice(), Slice(2,4)});
  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return a large result to make sure the solver does not fail
    return TensorDual::ones_like(x)*1.0e6;
  }
  auto x1delta = r.y.index({Slice(), Slice(2,3)})-x1f;
  auto x2delta = r.y.index({Slice(), Slice(3,4)})-x2f;
  auto Hf = hamiltonian_dual(xf, pf, W);
  TensorDual res = x*0.0;
  //The hamiltonian is zero at the terminal time 
  //because this is a minimum time problem
  res.index_put_({Slice(), 0}, x1delta);
  res.index_put_({Slice(), 1}, x2delta);
  res.index_put_({Slice(), 2}, Hf);
  std::cerr << "propagation result=";
  janus::print_dual(res);

  return res;

}

TensorDual jac_eval(const TensorDual& x, const TensorDual& params) {
  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int M = x.r.size(0);
  int D = 4;//Length of the state space vector in order [p1, p2, x1, x2] 
  int N = 5;//Length of the dual vector in order [p1, p2, x1, x2, tf]


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, x.r.options()), torch::zeros({M,D,N},x.r.options()));
  for (int i=0; i < M; i++) {
    y.index_put_({i, 0}, x.index({i, 0}));
    y.index_put_({i, 1}, x.index({i, 1}));
    y.index_put_({i, 2}, x10);
    y.index_put_({i, 3}, x20);
  }
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, x.r.options()), torch::zeros({M,2,N}, x.r.options()));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), 1}, x.index({Slice(), 2}));
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, x.r.options());
  options.AbsTol = torch::tensor({1e-6}, x.r.options());
  //Create an instance of the Radau5 class
  options.MaxNbrStep = 1000;
  auto tspanc = tspan.clone();
  auto yc = y.clone();
  auto paramsc = params.clone();

  janus::RadauTeD r(vdpdyns_ham, jac_ham, tspanc, yc, options, paramsc);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  int rescode = r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  TensorDual jacVal = TensorDual(torch::zeros({M, 3, 3}, torch::kFloat64),
                                  torch::zeros({M, 3, 3, N}, torch::kFloat64)); 

  if (rescode != 0) {
    std::cerr << "propagation failed\n";
    //Return default zeros.  These values will not be used
    return jacVal;
  }


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
  jacVal.index_put_({Slice(), 0, 0}, x1delta.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 0, 1}, x1delta.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 0, 2}, x1delta.index({Slice(), 0,N-1}));
  jacVal.index_put_({Slice(), 1, 0}, x2delta.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 1, 1}, x2delta.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 1, 2}, x2delta.index({Slice(), 0,N-1}));
  jacVal.index_put_({Slice(), 2, 0}, Hf.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 2, 1}, Hf.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 2, 2}, Hf.index({Slice(), 0,N-1}));
  std::cerr << "jacobian result=";
  janus::print_dual(jacVal);


  return jacVal;

}


//Create a main method for testing

int main(int argc, char *argv[])
{
  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  int M =1;
  int N =3;
  /*
  -28.8603   1.4887   6.1901*/

  torch::Tensor x0 = torch::ones({1, N}, torch::dtype(torch::kFloat64));
  for ( int i=0; i < M; i++) {
    x0.index_put_({i, 0}, p10+0.1*i);  //p1
    x0.index_put_({i, 1}, p20+0.1*i);  //p2
    x0.index_put_({i, 2}, ft);        //ft
  }
  torch::Tensor params = torch::zeros_like(x0);
  //Impose limits on the state space
  torch::Tensor xmin = torch::zeros_like(x0);
  xmin.index_put_({Slice(), 0}, -10000.0);
  xmin.index_put_({Slice(), 1}, -10000.0);
  xmin.index_put_({Slice(), 2}, 0.001);
  torch::Tensor xmax = torch::zeros_like(x0);
  xmax.index_put_({Slice(), 0}, 10000.0);
  xmax.index_put_({Slice(), 1}, 10000.0);
  xmax.index_put_({Slice(), 2}, 10.0);
  torch::Tensor dummy = torch::zeros({M, 1}, torch::kFloat64);
  auto x0d = TensorDual(x0.clone(), dummy.clone());
  auto paramsd = TensorDual(params.clone(), dummy.clone());
  auto xmind = TensorDual(xmin.clone(), dummy.clone());
  auto xmaxd = TensorDual(xmax.clone(), dummy.clone());

  
  auto res = newtTeD(x0d, paramsd, xmind, xmaxd, propagate, jac_eval);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  std::cerr << "roots=" << roots << "\n";
  std::cerr << "check=" << check << "\n";
  auto errors = Jfunc(propagate(roots, paramsd));
  std::cerr << "errors=" << errors << "\n";


  return 0;
}




