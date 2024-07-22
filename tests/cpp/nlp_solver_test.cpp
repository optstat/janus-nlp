#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/lnsrchte.hpp"
#include "../../src/cpp/newtte.hpp"
#include "../../src/cpp/lnsrchted.hpp"
#include "../../src/cpp/newtted.hpp"

/**
 * Function to reduce the vector function to a scalar function
 * returns a tensor of dimension [M]
*/
torch::Tensor J(torch::Tensor& x) {
    
    int M = x.size(0);
    if (x.dim() == 1)
    {
        return 0.5*x.square();
    }
    else
    {
        return 0.5*x.square().sum(1);
    }
}

TensorDual J(TensorDual& x) {
    
    int M = x.r.size(0);
    return 0.5*x.square().sum();
}


torch::Tensor cubic(torch::Tensor& x, torch::Tensor& params) {
    int M = x.size(0);

    return (x+1.0).pow(3);
}

TensorDual cubic_dual(TensorDual& x,TensorDual& params) {
    int M = x.r.size(0);

    return (x+1.0).pow(3);
}


// Gradient of the quadratic function
torch::Tensor cubic_gradient(torch::Tensor& x, torch::Tensor& params) {
    int M = x.size(0);
    return 3.0*(x+1.0).pow(2);
}

// Gradient of the quadratic function
TensorDual cubic_gradient_dual(TensorDual& x, TensorDual& params) {
    int M = x.r.size(0);
    return 3.0*(x+1.0).pow(2);
}


torch::Tensor func2d(const torch::Tensor& x, const torch::Tensor& params) {

    int M = x.size(0);
    auto y = torch::zeros({M, 2}, torch::dtype(torch::kFloat64));
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    y.index_put_({Slice(), 0}, x1.pow(2)+x2.pow(2)-4.0);
    y.index_put_({Slice(), 1}, x2+x1.exp()-1.0);
    return y;
}

TensorDual func2d_dual(const TensorDual& x, const TensorDual& params) {

    int M = x.r.size(0);
    auto y = TensorDual(torch::zeros({M, 2}, torch::dtype(torch::kFloat64)),
                        torch::zeros({M, 2, x.d.size(2)}, torch::dtype(torch::kFloat64)));    
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    y.index_put_({Slice(), 0}, x1.pow(2)+x2.pow(2)-4.0);
    y.index_put_({Slice(), 1}, x2+x1.exp()-1.0);
    return y;
}


torch::Tensor jac2d(const torch::Tensor& x, const torch::Tensor& params) 
{
    int M = x.size(0);
    torch::Tensor jac = torch::zeros({M, 2, 2}, torch::dtype(torch::kFloat64));
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    jac.index_put_({Slice(), 0, 0}, 2*x1);
    jac.index_put_({Slice(), 0, 1}, 2*x2);
    jac.index_put_({Slice(), 1, 0}, x1.exp());
    jac.index_put_({Slice(), 1, 1}, 1.0);
    return jac;
}

TensorMatDual jac2d_dual(const TensorDual& x, const TensorDual& params) 
{
    int M = x.r.size(0);
    TensorMatDual jac = TensorMatDual(torch::zeros({M, 2, 2}, torch::dtype(torch::kFloat64)),
                                      torch::zeros({M, 2, 2, x.d.size(2)}, torch::dtype(torch::kFloat64)));
    auto x1 = x.index({Slice(), 0});
    auto x2 = x.index({Slice(), 1});
    jac.index_put_({Slice(), 0, 0}, 2*x1);
    jac.index_put_({Slice(), 0, 1}, 2*x2);
    jac.index_put_({Slice(), 1, 0}, x1.exp());
    jac.index_put_({Slice(), 1, 1}, 1.0);
    return jac;
}




TEST(LineSearchTest, Cubic) 
{
    // Parameters for the quadratic function
    int M = 1;
    auto params = torch::zeros({M, 2}, torch::kDouble);

    // Initial guess
    torch::Tensor x0 = torch::ones({M, 2}, torch::kDouble);
    for (int i = 0; i < M; i++)
    {
        x0.index_put_({i, 0}, 0.0+0.01*i);
        x0.index_put_({i, 1}, 0.0+0.01*i);
    }
    torch::Tensor xmin = torch::ones({M, 2}, torch::kDouble)*-1000;
    torch::Tensor xmax = torch::ones({M, 2}, torch::kDouble)*1000;

    torch::Tensor stpmax = torch::ones({M}, torch::kDouble);
    torch::Tensor check = torch::zeros({M}, torch::kBool);

    // Function value and gradient at the initial guess
    torch::Tensor f0 = cubic(x0, params);
    auto J0 = J(f0);
    torch::Tensor g = cubic_gradient(x0, params);

    // Search direction (negative gradient)
    torch::Tensor p = -g;
    // Perform line search
    auto [x_new, J_new, p_new, check_new] = janus::lnsrchTe(x0, f0, J0, g, p, stpmax, params, xmin, xmax, cubic);

    // Print the results
    //std::cerr << (J_new < J0) << std::endl;
    EXPECT_TRUE((J_new < J0).all().item<bool>());
}



TEST(LineSearchTest, 2DFunction) {
    // Parameters for the quadratic function
    int M = 10;
    //create empty parameters
    auto params = torch::empty({M, 1}, torch::kDouble);

    // Initial guess
    torch::Tensor x0 = torch::ones({M, 2}, torch::kDouble);
    for (int i = 1; i <= M; i++)
    {
        x0.index_put_({i-1, 0}, 2.0);
        x0.index_put_({i-1, 1}, 2.0);
    }
    torch::Tensor xmin = torch::ones({M, 2}, torch::kDouble)*-1000;
    torch::Tensor xmax = torch::ones({M, 2}, torch::kDouble)*1000;


    torch::Tensor stpmax = torch::ones({M}, torch::kDouble);
    torch::Tensor check = torch::zeros({M}, torch::kBool);

    // Function value and gradient at the initial guess
    torch::Tensor f0 = func2d(x0, params);
    auto J0 = J(f0);
    torch::Tensor jac = jac2d(x0, params);
    torch::Tensor g = torch::einsum("mij, mi->mj", {jac, f0});
    //Use LU decomposition to calculate the search direction
    auto [LU, P] = janus::LUTe(jac);
    auto p = janus::solveluv(LU, P, -f0);

    // Perform line search
    auto [x_new, J_new, g_new, check_new] = janus::lnsrchTe(x0, f0, J0, g, p, stpmax, params,xmin, xmax, func2d);

    // Print the results
    /*std::cout << "Initial position: " << x0 << std::endl;
    std::cout << "Initial gradient: " << g << std::endl;
    std::cout << "Initial function value: " << f0 << std::endl;
    std::cout << "Initial error: " << janus::Jfunc(func2d(x0, params)) << std::endl;
    std::cout << "New position: " << x_new << std::endl;
    std::cout << "New gradient: " << g_new << std::endl;
    std::cout << "Check: " << check_new << std::endl;
    std::cout << "Solution error" << janus::Jfunc(func2d(x_new, params)) << std::endl;
    std::cout << "J0 = " << J0 << std::endl;
    std::cout << "J_new = " << J_new << std::endl;*/
    EXPECT_TRUE((J_new < J0).all().item<bool>());

}


TEST(LineSearchDualTest, Cubic) 
{
    // Parameters for the quadratic function
    int M = 1;
    int D = 2;
    // Initial guess
    TensorDual x0 = TensorDual(torch::ones({M, 2}, torch::kDouble),
                               torch::zeros({M, 2, D}, torch::kDouble));
    for (int i = 0; i < M; i++)
    {
        x0.index_put_({i, 0}, 0.0+0.01*i);
        x0.index_put_({i, 1}, 0.0+0.01*i);
    }
    TensorDual xmin = TensorDual(torch::ones({M, 2}, torch::kDouble)*-1000,
                                    torch::zeros({M, 2, D}, torch::kDouble));   
    TensorDual xmax = TensorDual(torch::ones({M, 2}, torch::kDouble)*1000,
                                    torch::zeros({M, 2, D}, torch::kDouble));   

    TensorDual stpmax = TensorDual(torch::ones({M,1}, torch::kDouble),
                                    torch::zeros({M, 1, D}, torch::kDouble));   
    torch::Tensor check = torch::zeros({M}, torch::kBool);
    auto params = TensorDual(torch::zeros({M, 2}, torch::dtype(torch::kFloat64)),
                             torch::zeros({M, 2, D}, torch::dtype(torch::kFloat64)));
    // Function value and gradient at the initial guess
    auto f0 = cubic_dual(x0, params);
    auto J0 = J(f0);
    auto g = cubic_gradient_dual(x0, params);

    // Search direction (negative gradient)
    TensorDual p = -g;
    // Perform line search
    auto [x_new, J_new, p_new, check_new] = janus::lnsrchTeD(x0, 
                                                             f0, 
                                                             J0, 
                                                             g, 
                                                             p, 
                                                             stpmax, 
                                                             params, 
                                                             xmin, 
                                                             xmax, 
                                                             cubic_dual);

    // Print the results
    //std::cerr << (J_new < J0) << std::endl;
    EXPECT_TRUE((J_new < J0).all().item<bool>());
}




torch::Tensor func(const torch::Tensor& x, const torch::Tensor& params) 
{
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


TEST(NewtGlobalTest, 2DFunction) {
  int M =10;
  torch::Tensor x0 = torch::ones({M, 2}, torch::dtype(torch::kFloat64));
  for ( int i=0; i < M; i++) {
    x0.index_put_({i, 0}, 2.0+0.1*i);
    x0.index_put_({i, 1}, 1.0+0.1*i);
  }
  torch::Tensor xmin = torch::ones({M, 2}, torch::kDouble)*-1000;
  torch::Tensor xmax = torch::ones({M, 2}, torch::kDouble)*1000;

  torch::Tensor params = torch::zeros({M, 2}, torch::dtype(torch::kFloat64));
  auto res = newtTe(x0, params, xmin, xmax, func, jac);
  auto roots = std::get<0>(res);
  auto check = std::get<1>(res);
  auto errors = Jfunc(func(roots, params));
  auto sols = torch::zeros_like(errors);
  
  //std::cout << "Root=" << roots << std::endl;
  //std::cout << "Check=" << check << std::endl;
  //std::cout << "Error=" << Jfunc(func(roots, params)) << std::endl;
  EXPECT_TRUE(torch::allclose(errors, sols));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
