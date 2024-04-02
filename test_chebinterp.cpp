#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include "chebinterp.hpp"
#include <complex>

#include <tbb/parallel_for.h>

int main()
{
    const int N = 10;
    const int dim = 2;

    typedef std::complex<double> T;
    Eigen::Array<double, dim, Eigen::Dynamic> points = Eigen::Array<double, dim, Eigen::Dynamic, Eigen::RowMajor>::Random(dim, N)+0.01;

    const double H = 0.1;
    const double k = 1;
    typedef Eigen::Vector<double, dim> Point;
    const Point xc{0, H / 2.0};
    auto kernel = [&](auto pnt) {
        auto d=pnt.matrix().squaredNorm();
	return exp(d)/(d+1);
    };

    const int p = 15;
    Eigen::Vector<int, 2> ns{p-2,p};
    Eigen::Array<double, dim, Eigen::Dynamic> interp_nodes = ChebychevInterpolation::chebnodesNd<double,-1,-1>(ns);

    Eigen::Array<T, Eigen::Dynamic, 1> interp_values(interp_nodes.cols());
    interp_values.fill(0);
    for (int i = 0; i < interp_nodes.cols(); i++) {
        const Eigen::Vector<double, dim> &pnt = interp_nodes.col(i);

        interp_values[i] = kernel(pnt);
    }
    std::cout<<"1"<<std::endl;

    Eigen::Array<T, Eigen::Dynamic, 1> values(N);
    for (int i = 0; i < values.size(); i++) {
        values[i] = kernel(points.col(i));
    }
    std::cout<<"2"<<std::endl;
    Eigen::Array<T, Eigen::Dynamic, 1> approx_values(N);
    //Eigen::internal::set_is_malloc_allowed(false);

    ChebychevInterpolation::parallel_evaluate<T, dim,1>(points, interp_values, approx_values,p);

    //Eigen::internal::set_is_malloc_allowed(true);

    approx_values -= values;

    std::cout << "error=" << approx_values.matrix().norm() / sqrt(N) << std::endl;

    return 0;

}
