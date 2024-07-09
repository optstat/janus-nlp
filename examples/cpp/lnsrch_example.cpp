#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <functional>
#include "../../src/cpp/lnsrchTe.hpp"
#include <janus/lute.hpp>
#include <janus/janus_util.hpp>
using Slice = torch::indexing::Slice;

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

torch::Tensor cubic(torch::Tensor& x, torch::Tensor& params) {
    int M = x.size(0);

    return (x+1.0).pow(3);
}

// Gradient of the quadratic function
torch::Tensor cubic_gradient(torch::Tensor& x, torch::Tensor& params) {
    int M = x.size(0);
    return 3.0*(x+1.0).pow(2);
}


// Quadratic function
torch::Tensor quadratic(torch::Tensor& x, torch::Tensor& params) {
    torch::Tensor A = params.index({Slice(), Slice(0, 4)}).reshape({-1, 2, 2});
    torch::Tensor b = params.index({Slice(), Slice(4, 6)});
    auto Ax =torch::einsum("mij, mj->mi", {A,x});
    auto xAx = torch::einsum("mi, mi->m", {x,Ax});
    auto bx = torch::einsum("mi, mi->m", {b,x});
    return 0.5 *xAx - bx;
}

// Gradient of the quadratic function
torch::Tensor quadratic_gradient(torch::Tensor& x, torch::Tensor& params) {
    torch::Tensor A = params.index({Slice(), Slice(0, 4)}).reshape({-1, 2, 2});
    torch::Tensor b = params.index({Slice(), Slice(4, 6)});
    auto Ax = torch::einsum("mij, mj->mi", {A, x});
    return Ax - b;
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

torch::Tensor jac2d(const torch::Tensor& x, const torch::Tensor& params) {
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

void test_quadratic()
{
    // Parameters for the quadratic function
    int M = 2;
    torch::Tensor A = torch::zeros({M, 2, 2}, torch::kDouble);
    A.index_put_({Slice(), 0, 0}, 3.0);
    A.index_put_({Slice(), 0, 1}, 2.0);
    A.index_put_({Slice(), 1, 0}, 2.0);
    A.index_put_({Slice(), 1, 1}, 6.0);
    torch::Tensor b = torch::zeros({M, 2}, torch::kDouble);
    b.index_put_({Slice(), 0}, 2.0);
    b.index_put_({Slice(), 1}, -8.0);
    auto params = torch::cat({A.flatten({1}), b}, 1);

    // Initial guess
    torch::Tensor x0 = torch::ones({M, 2}, torch::kDouble);
    for (int i = 0; i < M; i++)
    {
        x0.index_put_({i, 0}, 0.0+0.01*i);
        x0.index_put_({i, 1}, 0.0+0.01*i);
    }

    torch::Tensor stpmax = torch::ones({M}, torch::kDouble);
    torch::Tensor check = torch::zeros({M}, torch::kBool);

    // Function value and gradient at the initial guess
    torch::Tensor f0 = quadratic(x0, params);
    auto J0 = J(f0);
    torch::Tensor g = quadratic_gradient(x0, params);

    // Search direction (negative gradient)
    torch::Tensor p = -g;
    // Perform line search
    auto [x_new, J_new, p_new, check_new] = janus::lnsrchTe(x0, f0, J0, g, p, stpmax, params, quadratic);

    // Print the results
    std::cout << "Initial position: " << x0 << std::endl;
    std::cout << "Initial gradient: " << g << std::endl;
    std::cout << "Initial function value: " << f0 << std::endl;
    std::cout << "New position: " << x_new << std::endl;
    std::cout << "New direction: " << p_new << std::endl;
    std::cout << "Check: " << check_new << std::endl;

}

void test_cubic()
{
    // Parameters for the quadratic function
    int M = 1;
    //create empty parameters
    auto params = torch::empty({M, 1}, torch::kDouble);

    // Initial guess
    torch::Tensor x0 = torch::zeros({M, 1}, torch::kDouble);
    for (int i = 1; i <= M; i++)
    {
        x0.index_put_({i-1, 0}, 0.0);
    }

    torch::Tensor stpmax = torch::ones({M}, torch::kDouble);
    torch::Tensor check = torch::zeros({M}, torch::kBool);

    // Function value and gradient at the initial guess
    torch::Tensor f0 = cubic(x0, params);
    auto J0 = J(f0);
    torch::Tensor g = cubic_gradient(x0, params);

    // Search direction (negative gradient)
    torch::Tensor p = -g;
    // Perform line search
    auto [x_new, J_new, g_new, check_new] = janus::lnsrchTe(x0, f0, J0, g, p, stpmax, params, cubic);

    // Print the results
    std::cout << "Initial position: " << x0 << std::endl;
    std::cout << "Initial gradient: " << g << std::endl;
    std::cout << "Initial function value: " << f0 << std::endl;
    std::cout << "New position: " << x_new << std::endl;
    std::cout << "New gradient: " << g_new << std::endl;
    std::cout << "Check: " << check_new << std::endl;

}

void test_2d()
{
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

    torch::Tensor stpmax = torch::ones({M}, torch::kDouble);
    torch::Tensor check = torch::zeros({M}, torch::kBool);

    // Function value and gradient at the initial guess
    torch::Tensor f0 = func2d(x0, params);
    auto J0 = J(f0);
    torch::Tensor jac = jac2d(x0, params);
    torch::Tensor g = torch::einsum("mij, mi->mj", {jac, f0});
    //Use LU decomposition to calculate the search direction
    auto [LU, P] = janus::lu(jac);
    auto p = janus::solveluv(LU, P, -f0);

    // Perform line search
    auto [x_new, J_new, g_new, check_new] = janus::lnsrchTe(x0, f0, J0, g, p, stpmax, params, func2d);

    // Print the results
    std::cout << "Initial position: " << x0 << std::endl;
    std::cout << "Initial gradient: " << g << std::endl;
    std::cout << "Initial function value: " << f0 << std::endl;
    std::cout << "Initial error: " << janus::Jfunc(func2d(x0, params)) << std::endl;
    std::cout << "New position: " << x_new << std::endl;
    std::cout << "New gradient: " << g_new << std::endl;
    std::cout << "Check: " << check_new << std::endl;
    std::cout << "Solution error" << janus::Jfunc(func2d(x_new, params)) << std::endl;

}

int main(int argc, char const *argv[])
{
    test_2d();
    return 0;
}