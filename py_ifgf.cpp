#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <iostream>

#include <tbb/global_control.h>

#include "helmholtz_ifgf.hpp"

namespace py = pybind11;


#include <fenv.h>
PYBIND11_MODULE(pyifgf, m) {
    m.doc() = R"pbdoc(
        A fast library implementing the Inetpolated Factored Greens function
    )pbdoc";

    //feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    /*const int num_threads= std::atoi(std::getenv("IFGF_NUM_THREADS"));
    std::cout<<"running on "<<num_threads<<" threads"<<std::endl;
    auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,   num_threads );
    */
    typedef HelmholtzIfgfOperator<3> OpType;
    py::class_< OpType>(m,"HelmholtzIfgfOperator")
	.def(py::init<std::complex<double>, int,size_t,int>())
	.def("mult", &OpType::mult)	     
	.def("init", &OpType::init);	     
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

