#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "newt.hpp"
using Slice = torch::indexing::Slice;



torch::Tensor func(const torch::Tensor& x, const torch::Tensor& params) {

    int M = x.size(0);
    auto y = torch::zeros({M, 2}, torch::dtype(torch::kFloat64));
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    y.index_put_({Slice(), 0}, x2.pow(2)+x1.pow(2)-4.0);
    y.index_put_({Slice(), 1}, x2-x1.exp());
    return y;
}

torch::Tensor jac(const torch::Tensor& x, const torch::Tensor& params) {
    int M = x.size(0);
    torch::Tensor jac = torch::zeros({M, 2, 2}, torch::dtype(torch::kFloat64));
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    jac.index_put_({Slice(), 0, 0}, 2*x1);
    jac.index_put_({Slice(), 0, 1}, 2*x2);
    jac.index_put_({Slice(), 1, 0}, -x1.exp());
    jac.index_put_({Slice(), 1, 1}, 1.0);
    return jac;
}



using namespace janus;
int main() {
  int M =1000;
  torch::Tensor x0 = torch::ones({M, 2}, torch::dtype(torch::kFloat64));
  for ( int i=0; i < M; i++) {
    x0.index_put_({i, 0}, 2.0+0.1*i);
    x0.index_put_({i, 1}, 1.0+0.1*i);
  }
  torch::Tensor params = torch::zeros({M, 2}, torch::dtype(torch::kFloat64));
  auto res = newtTe(x0, params, func, jac);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  std::cout << "Root=" << roots << std::endl;
  std::cout << "Check=" << check << std::endl;
  std::cout << "Error=" << Jfunc(func(roots, params)) << std::endl;
}