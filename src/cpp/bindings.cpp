#define HEADER_ONLY
#include <pybind11/pybind11.h>
//#include "lnsrchte.hpp"
//#include "lnsrchted.hpp"
//#include "newtte.hpp"
//#include "newtted.hpp"
//#include "../../examples/cpp/lnsrchte_example.hpp"
//#include "../../examples/cpp/newted_control_example.hpp"
//#include "../../examples/cpp/newted_vdp_example.hpp"
#include "../../examples/cpp/mint_vdp_example.hpp"
#include "../../examples/cpp/mint_auglang_vdp_example.hpp"
#include "../../examples/cpp/minu_auglang_linear_example.hpp"

namespace py = pybind11;

PYBIND11_MODULE(janus_nlp, m) {

    // Expose additional functions or classes if needed
    //m.def("lnsrchTe",  &janus::nlp::lnsrchTe, "Tensor based line search");
    //m.def("lnsrchTeD", &janus::nlp::lnsrchTe, "Line Search using extend dual tensors");
    //m.def("newtTe", &janus::nlp::newtTe, "Global Newton method for BVP");
    //m.def("newtTeD", &janus::nlp::newtTeD, "Global Newton method for BVP using dual tensors");
    //m.def("test_2d", &janus::nlp::examples::test_2d, "Test 2D function");
    //m.def("vdpNewt", &janus::nlp::examples::vdp::vdpNewt, "Control function for the VDPC example");
    //m.def("set_x0", &janus::nlp::examples::vdp::set_x0, "Set the initial state for the VDPC example");
    //m.def("set_xf", &janus::nlp::examples::vdp::set_xf, "Set the final state for the VDPC example");
    //m.def("set_mu", &janus::nlp::examples::vdp::set_mu, "Set the regularization weight for the VDPC example");
    //m.def("vdp_solve", &janus::nlp::examples::vdp::vdp_solve, "Solve the VDPC example");
    ///m.def("vdp_solve_traj", &janus::nlp::examples::vdp::solve_traj, "Generate costate and state trajectories");
    //m.def("vdpNewt", &janus::nlp::examples::vdp::vdpNewt, "Newton method for the VDPC example");
    m.def("set_mint_xf", &janus::nlp::examples::vdp::mint::set_xf, "Set the final point for the mint example");
    m.def("set_mint_x0", &janus::nlp::examples::vdp::mint::set_x0, "Set the initial point for the mint example");
    m.def("mint_vdp_solve", &janus::nlp::examples::vdp::mint::mint_vdp_solve, "Solve the VDPC example");
    m.def("mint_jac_eval", &janus::nlp::examples::vdp::mint::mint_jac_eval, "Evaluate the Jacobian of the mint example");
    m.def("mint_jac_eval_fd", &janus::nlp::examples::vdp::mint::mint_jac_eval_fd, "Evaluate the Jacobian of the mint example");
    m.def("mint_set_mu", &janus::nlp::examples::vdp::mint::set_mu, "Set the regularization weight for the mint example");
    m.def("mint_set_W", &janus::nlp::examples::vdp::mint::set_W, "Set the final state for the mint example");
    m.def("mint_auglangr_propagate", &janus::nlp::examples::vdp::mint::auglang::mint_auglangr_propagate, "Augmented Lagrangian and its gradients for the mint example");
    m.def("mint_propagate", &janus::nlp::examples::vdp::mint::auglang::mint_propagate, "Objective constraint and its jacobian for the mint example");
    m.def("set_auglangr_xf", &janus::nlp::examples::vdp::mint::auglang::set_xf, "Set the final point for the mint example");
    m.def("set_auglangr_x0", &janus::nlp::examples::vdp::mint::auglang::set_x0, "Set the initial point for the mint example");
    m.def("calc_ustar", &janus::nlp::examples::vdp::mint::auglang::calc_ustar, "Calculate the control for the mint example");
    m.def("set_ulimits", &janus::nlp::examples::vdp::mint::auglang::set_ulimits, "Set the control limits for the mint example");
    m.def("linear_minu_set_xf", &janus::nlp::examples::linear::minu::auglang::set_xf, "Set the final point for the linear minu example");
    m.def("linear_minu_set_x0", &janus::nlp::examples::linear::minu::auglang::set_x0, "Set the initial point for the linear minu example");
    m.def("linear_minu_set_a", &janus::nlp::examples::linear::minu::auglang::set_a, "Set the constant a for the linear minu example");
    m.def("linear_minu_set_b", &janus::nlp::examples::linear::minu::auglang::set_b, "Set the constant b for the linear minu example");
    m.def("linear_minu_set_ft", &janus::nlp::examples::linear::minu::auglang::set_ft, "Set the constant b for the linear minu example");
    m.def("linear_minu_auglangr_propagate", &janus::nlp::examples::linear::minu::auglang::minu_auglangr_propagate, "Augmented Lagrangian and its gradients for the linear minu example");
}
