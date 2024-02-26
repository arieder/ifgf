#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"

const int dim=3;

const std::complex<double>  k = std::complex<double>(0, 10);
typedef Eigen::Vector<double,dim> Point;
std::complex<double> kernel(const Point& x, const Point& y)
{
    double d = (x - y).norm();

    return d == 0 ? 0 : (1 / (4 * M_PI)) * exp(-k * d) / d;
}


#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>


int main()
{
    
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray ;

    const int N = 1000000;


    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
    //oneapi::tbb::task_arena arena(1);

    HelmholtzIfgfOperator<dim> op(k,1000,15);

    PointArray srcs = (PointArray::Random(dim,N).array());
    PointArray targets = (PointArray::Random(dim, N).array());

    op.init(srcs, targets);

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> weights(N);
    weights = Eigen::VectorXd::Random(N);

    feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    std::cout<<"mult"<<std::endl;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> result = op.mult(weights);
    std::cout << "done multiplying" << std::endl;

    srand((unsigned) time(NULL));
    double maxE = 0;
    for (int j = 0; j < 100; j++) {
        std::complex<double> val = 0;
        int index = rand() % targets.cols();
        //std::cout<<"idx"<<index<<std::endl;
        for (int i = 0; i < srcs.cols(); i++) {
            val += weights[i] * kernel(srcs.col(i), targets.col(index));
        }

        double e = std::abs(val - result[index]);
        maxE = std::max(e, maxE);
        //std::cout<<"e="<<e<<" val="<<val<<" vs" <<result[index]<<std::endl;
    }

    std::cout << "summary: e=" << maxE << std::endl;

}
