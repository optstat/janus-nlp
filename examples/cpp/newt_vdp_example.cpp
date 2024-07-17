/**
 * Use the Van Der Pol oscillator as an example
 * To calculate optimal control for minimum time
 */
#include <janus/radauted.hpp>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include <janus/janus_ode_common.hpp>
#include "../../src/cpp/lnsrchTe.hpp"
#include "../../src/cpp/newtTe.hpp"
#include "matplotlibcpp.h"

using namespace janus;
namespace plt = matplotlibcpp;


/**
 * Radau example using the Van der Pol oscillator 
 * Using the Hamiltonian with dual number approach to calcuate the dynamics and
 * the Jacobian
*/
using Slice = torch::indexing::Slice;
double W = 0.1;
double x1f = 1.0;
double x2f = -1.75;
double x10 = 2.0;
double x20 = 0.0;
double ft = 1.0;
double mu=1.0;

 

TensorDual control_dual(const TensorDual& x1, 
          const TensorDual& x2,
          const TensorDual& p1,
          const TensorDual& p2, 
          double W=1.0) {
  auto u = -p2*((1-x1*x1)*x2-x1)/W;
  auto m = u < 0.01;

  if (m.any().item<bool>()) {
    u.index_put_({m}, 0.01);
  } 
  return u; //Return the through copy elision
}


torch::Tensor control(const torch::Tensor& x1, 
          const torch::Tensor& x2,
          const torch::Tensor& p1,
          const torch::Tensor& p2, 
          double W=1.0) 
{
  auto u = -p2*((1-x1*x1)*x2-x1)/W;
  auto m = u < 0.01;
  if (m.any().item<bool>()) {
    u.index_put_({m}, 0.01);
  }

  return u; //Return the through copy elision
}



TensorDual hamiltonian_dual(const TensorDual& x, 
                            const TensorDual& p, 
                            double W) {
  TensorDual p1 = p.index({Slice(), 0});  
  TensorDual p2 = p.index({Slice(), 1});  
  TensorDual x1 = x.index({Slice(), 0});  
  TensorDual x2 = x.index({Slice(), 1});  
  auto H = p1*x2+p2*mu*(1-x1*x1)*x2-p2*mu*x1-p2*p2/W+p2*p2/(2*W)+1; //Return the through copy elision
  return H; //Return the through copy elision
}


torch::Tensor hamiltonian(const torch::Tensor& x, 
                          const torch::Tensor& p, 
                   double W) {
  torch::Tensor p1 = p.index({Slice(), 0});  
  torch::Tensor p2 = p.index({Slice(), 1});  
  torch::Tensor x1 = x.index({Slice(), 0});  
  torch::Tensor x2 = x.index({Slice(), 1});  
  auto H = p1*x2+p2*mu*(1-x1*x1)*x2-p2*mu*x1-p2*p2/W+p2*p2/(2*W)+1; //Return the through copy elision
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
TensorDual vdpdyns(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  auto dyns= y*0.0;
  auto p1 = y.index({Slice(), 0});
  auto p2 = y.index({Slice(), 1});
  auto x1 = y.index({Slice(), 2});
  auto x2 = y.index({Slice(), 3});
  dyns.index_put_({Slice(), 0}, -2*p2*mu*x1*x2);
  dyns.index_put_({Slice(), 1}, p1+p2*mu*(1-x1*x1));
  dyns.index_put_({Slice(), 2}, x2);
  dyns.index_put_({Slice(), 3}, mu*((1-x1*x1)*x2-x1) -p2/W);
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

TensorMatDual jac(const TensorDual& t, 
                   const TensorDual& y, 
                   const TensorDual& params) {
  auto jac = TensorMatDual(torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1)}, torch::kFloat64), 
                           torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1), y.d.size(2)}, torch::kFloat64));
  auto p1 = y.index({Slice(), 0});
  auto p2 = y.index({Slice(), 1});
  auto x1 = y.index({Slice(), 2});
  auto x2 = y.index({Slice(), 3});
  jac.index_put_({Slice(), 0, 1}, -2*mu*x1*x2-mu);
  jac.index_put_({Slice(), 0, 2}, -2*mu*p2*x2);
  jac.index_put_({Slice(), 0, 3}, -2*mu*p2*x1); 

  TensorDual one = TensorDual(torch::ones_like(p1.r), torch::zeros_like(p1.d));
  TensorDual zero = TensorDual(torch::zeros_like(p1.r), torch::zeros_like(p1.d));
  jac.index_put_({Slice(), 1, 0}, one);
  jac.index_put_({Slice(), 1, 1}, mu*(1-x1*x1));
  jac.index_put_({Slice(), 1, 2}, -2*mu*p2*x1);
  jac.index_put_({Slice(), 1, 3}, zero);

  jac.index_put_({Slice(), 2, 1}, -one/W);
  jac.index_put_({Slice(), 2, 2}, mu*(-2*x1*x2-one));
  jac.index_put_({Slice(), 2, 3}, mu*(1-x1*x1));
  

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
  int D = 3; //Number of variables
  int N = 5; //Length of the dual vector in order [p1, p2, x1, x2, tf]
  auto device = x.device();

  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, 4}, torch::kF64).to(device), torch::zeros({M,4,N}, torch::kF64).to(device));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, x.index({i, 0}));
    y.r.index_put_({i, 1}, x.index({i, 1}));
    y.r.index_put_({i, 2}, x10);
    y.r.index_put_({i, 3}, x20);
    y.d.index_put_({i, 0, 0}, 1.0);
    y.d.index_put_({i, 1, 1}, 1.0);
    y.d.index_put_({i, 2, 2}, 1.0);
    y.d.index_put_({i, 3, 3}, 1.0);
  }
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, x.index({Slice(), 2}));
  tspan.d.index_put_({Slice(), 1, N-1}, 1.0); //Sensitivity to the final times
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-6}, torch::kFloat64).to(device);


  auto params_dual = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));
  //Create an instance of the Radau5 class
  //Call the solve method of the Radau5 class
  janus::RadauTeD r(vdpdyns, jac, tspan, y, options, params_dual);   // Pass the correct arguments to the constructor`
  r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);
  int nrows = r.yout.r.size(1);
  auto pf = r.y.index({Slice(), Slice(0,2)});
  auto xf = r.y.index({Slice(), Slice(2,4)});
  
  auto x1delta = r.y.index({Slice(), Slice(2,3)})-x1f;
  auto x2delta = r.y.index({Slice(), Slice(3,4)})-x2f;
  auto Hf = hamiltonian_dual(xf, pf, W);
  torch::Tensor res = torch::zeros({M, 3}, torch::kFloat64);
  //The hamiltonian is zero at the terminal time 
  //because this is a minimum time problem
  res.index_put_({Slice(), 0}, x1delta.r);
  res.index_put_({Slice(), 1}, x2delta.r);
  res.index_put_({Slice(), 2}, Hf.r);
  std::cerr << "propagation result=";
  janus::print_tensor(res);

  return res;

}

torch::Tensor jac_eval(const torch::Tensor& x, const torch::Tensor& params) {
  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int M = x.size(0);
  int D = 4;//Length of the state space vector in order [p1, p2, x1, x2] 
  int N = 5;//Length of the dual vector in order [p1, p2, x1, x2, tf]


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, x.options()), torch::zeros({M,D,N},x.options()));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, x.index({i, 0}));
    y.r.index_put_({i, 1}, x.index({i, 1}));
    y.r.index_put_({i, 2}, x10);
    y.r.index_put_({i, 3}, x20);
    y.d.index_put_({i, 0, 0}, 1.0);
    y.d.index_put_({i, 1, 1}, 1.0);
    y.d.index_put_({i, 2, 2}, 1.0);
    y.d.index_put_({i, 3, 3}, 1.0);
  }
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, x.options()), torch::zeros({M,2,N}, x.options()));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, x.index({Slice(), 2}));
  tspan.d.index_put_({Slice(), 1, N-1}, 1.0); //Sensitivity to the final times
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-3}, x.options());
  options.AbsTol = torch::tensor({1e-6}, x.options());
  //Create an instance of the Radau5 class
  auto params_dual = TensorDual(params.clone(), torch::zeros({M,params.size(1),N}));

  janus::RadauTeD r(vdpdyns, jac, tspan, y, options, params_dual);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();
  //std::cerr << "r.Last=";
  //std::cerr << r.Last << "\n";
  //std::cerr << "r.h=";
  //janus::print_dual(r.h);

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
  torch::Tensor jacVal = torch::zeros({M, 3, 3}, torch::kFloat64);
  jacVal.index_put_({Slice(), 0, 0}, x1delta.d.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 0, 1}, x1delta.d.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 0, 2}, x1delta.d.index({Slice(), 0,N-1}));
  jacVal.index_put_({Slice(), 1, 0}, x2delta.d.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 1, 1}, x2delta.d.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 1, 2}, x2delta.d.index({Slice(), 0,N-1}));
  jacVal.index_put_({Slice(), 2, 0}, Hf.d.index({Slice(), 0,0}));
  jacVal.index_put_({Slice(), 2, 1}, Hf.d.index({Slice(), 0,1}));
  jacVal.index_put_({Slice(), 2, 2}, Hf.d.index({Slice(), 0,N-1}));
  std::cerr << "jacobian result=";
  janus::print_tensor(jacVal);


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

  torch::Tensor x0 = torch::ones({1, N}, torch::dtype(torch::kFloat64));
  for ( int i=0; i < M; i++) {
    x0.index_put_({i, 0}, -5.0+0.1*i);  //p1
    x0.index_put_({i, 1}, 6.0+0.1*i);  //p2
    x0.index_put_({i, 2}, ft);        //x1
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

  
  auto res = newtTe(x0, params, xmin, xmax, propagate, jac_eval);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  std::cerr << "roots=" << roots << "\n";
  std::cerr << "check=" << check << "\n";
  auto errors = Jfunc(propagate(roots, params));
  std::cerr << "errors=" << errors << "\n";


  return 0;
}




