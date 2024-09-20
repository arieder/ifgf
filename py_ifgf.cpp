#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <iostream>

#include <tbb/global_control.h>

#include "double_layer_helmholtz_ifgf.hpp"
#include "helmholtz_ifgf.hpp"
#include "modified_helmholtz_ifgf.hpp"
#include "grad_helmholtz_ifgf.hpp"
#include "laplace_ifgf.hpp"

namespace py = pybind11;

template< typename OpType,typename T>
auto addOp(auto& m, const char* name)
{
   return py::class_< OpType>(m,name)
       .def(py::init<T, int,size_t,int,double>())
       .def("mult", &OpType::mult)	     
       .def("init", &OpType::init);	     
}

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
    addOp<ModifiedHelmholtzIfgfOperator<3>,std::complex<double> >(m,"MofifiedHelmholtzIfgfOperator");
    addOp<GradHelmholtzIfgfOperator<3>,std::complex<double> >(m,"GradHelmholtzIfgfOperator")
	.def("setDx", &GradHelmholtzIfgfOperator<3>::setDx);

    addOp<DoubleLayerHelmholtzIfgfOperator<3>,std::complex<double> >(m,"DoubleLayerHelmholtzIfgfOperator");


    py::class_< LaplaceIfgfOperator<3>>(m,"LaplaceIfgfOperator")
	.def(py::init<int,size_t,int,double>())
       .def("mult", &LaplaceIfgfOperator<3>::mult)	     
       .def("init", &LaplaceIfgfOperator<3>::init);	     


    //addOp<GradHelmholtzIfgfOperator<3,1>,std::complex<double> >(m,"HelmholtzDyIfgfOperator");
    //addOp<GradHelmholtzIfgfOperator<3,2>,std::complex<double> >(m,"HelmholtzDzIfgfOperator");
    
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

