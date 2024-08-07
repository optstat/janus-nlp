#include <pybind11/pybind11.h>
#include "lnsrchte.hpp"
#include "lnsrchted.hpp"
#include "newtte.hpp"
#include "newtted.hpp"
#include "../../examples/cpp/lnsrchte_example.hpp"
#include "../../examples/cpp/newted_control_example.hpp"
#include "../../examples/cpp/newted_vdp_example.hpp"

namespace py = pybind11;

PYBIND11_MODULE(janus_nlp, m) {

    // Expose additional functions or classes if needed
    m.def("lnsrchTe",  &janus::nlp::lnsrchTe, "Tensor based line search");
    m.def("lnsrchTeD", &janus::nlp::lnsrchTe, "Line Search using extend dual tensors");
    m.def("newtTe", &janus::nlp::newtTe, "Global Newton method for BVP");
    m.def("newtTeD", &janus::nlp::newtTeD, "Global Newton method for BVP using dual tensors");
    m.def("test_2d", &janus::nlp::examples::test_2d, "Test 2D function");
    m.def("vdpNewt", &janus::nlp::examples::vdp::vdpNewt, "Control function for the VDPC example");
    m.def("set_x0", &janus::nlp::examples::vdp::set_x0, "Set the initial state for the VDPC example");
    m.def("set_xf", &janus::nlp::examples::vdp::set_xf, "Set the final state for the VDPC example");
    m.def("set_mu", &janus::nlp::examples::vdp::set_mu, "Set the regularization weight for the VDPC example");
    m.def("calc_p10_tensor", 
          static_cast<torch::Tensor (*)(const torch::Tensor &)>(&janus::nlp::examples::vdp::calc_p10), 
          "Calculate the initial guess for the first Lagrange multiplier with torch::Tensor");
    m.def("calc_p10_tensor_dual", 
          static_cast<TensorDual (*)(const TensorDual &)>(&janus::nlp::examples::vdp::calc_p10), 
          "Calculate the initial guess for the first Lagrange multiplier with TensorDual");
    m.def("vdp_solve", &janus::nlp::examples::vdp::vdp_solve, "Solve the VDPC example");
    m.def("vdp_solve_traj", &janus::nlp::examples::vdp::solve_traj, "Generate costate and state trajectories");
    m.def("vdpNewt", &janus::nlp::examples::vdp::vdpNewt, "Newton method for the VDPC example");
}
