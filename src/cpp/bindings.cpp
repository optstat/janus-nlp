#include <pybind11/pybind11.h>
#include "lnsrchte.hpp"
#include "lnsrchted.hpp"
#include "newtte.hpp"
#include "newtted.hpp"
#include "../../examples/cpp/lnsrchte_example.hpp"
#include "../../examples/cpp/newted_control_example.hpp"

namespace py = pybind11;

PYBIND11_MODULE(janus_nlp, m) {

    // Expose additional functions or classes if needed
    m.def("lnsrchTe",  &janus::nlp::lnsrchTe, "Tensor based line search");
    m.def("lnsrchTeD", &janus::nlp::lnsrchTe, "Line Search using extend dual tensors");
    m.def("newtTe", &janus::nlp::newtTe, "Global Newton method for BVP");
    m.def("newtTeD", &janus::nlp::newtTeD, "Global Newton method for BVP using dual tensors");
    m.def("test_2d", &janus::nlp::examples::test_2d, "Test 2D function");
    m.def("vdpc_solve", &janus::nlp::examples::vdpc::solve, "Control function for the VDPC example");
}
