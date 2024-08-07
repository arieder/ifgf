#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"

#include "combined_field_helmholtz_ifgf.hpp"

const int dim=3;

typedef std::complex<double> Complex;
const double  kappa = 10;
typedef Eigen::Vector<double,dim> Point;
std::complex<double> kernel(const Point& x, const Point& y, const Point& normal)
{    
    double norm = (x-y).norm();
    double nxy = -normal.dot(x-y);
    auto kern = exp(Complex(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
    // return kern;

    if(norm >1e-12) return kern;
    else return 0;
}


#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>



int main()
{
    
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray ;

    const int N = 100000;


    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
    //oneapi::tbb::task_arena arena(1);

    CombinedFieldHelmholtzIfgfOperator<dim> op(kappa,250,10,2,-1e-8);

    PointArray srcs = (PointArray::Random(dim,N).array());
    PointArray normals = (PointArray::Random(dim,N).array());
    PointArray targets = (PointArray::Random(dim, N).array());



    //feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    op.init(srcs, targets,normals);

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> weights(N);
    weights = Eigen::VectorXd::Random(N);

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> result;
    for(int i=0;i<10;i++) {
	std::cout<<"mult"<<std::endl;
	result = op.mult(weights);
	std::cout << "done multiplying" << std::endl;
    }

    srand((unsigned) time(NULL));
    double maxE = 0;
    for (int j = 0; j < 100; j++) {
        std::complex<double> val = 0;
        int index = rand() % targets.cols();
        //std::cout<<"idx"<<index<<std::endl;
        for (int i = 0; i < srcs.cols(); i++) {
            val += weights[i] * kernel(srcs.col(i), targets.col(index),normals.col(i));
        }

        double e = std::abs(val - result[index]);
        maxE = std::max(e, maxE);
        //std::cout<<"e="<<e<<" val="<<val<<" vs" <<result[index]<<std::endl;
    }

    std::cout << "summary: e=" << maxE << std::endl;

}
