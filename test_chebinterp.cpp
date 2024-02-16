#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include "chebinterp.hpp"
#include <complex>

#include <tbb/parallel_for.h>

int main()
{
    const int N = 500;
    const int dim = 3;

    typedef std::complex<double> T;
    Eigen::Array<double, dim, Eigen::Dynamic, Eigen::RowMajor> points = Eigen::Array<double, dim, Eigen::Dynamic, Eigen::RowMajor>::Random(dim, N);

    const double H = 1;
    const double k = 1;
    typedef Eigen::Vector<double, dim> Point;
    const Point xc{0, H / 2,0};
    auto kernel = [&](auto pnt) {
        //pnt[0]=r , pnt[1]=theta
        Point x;
        x.fill(0);
        x[0] = pnt[0]*cos(pnt[1]);
        x[1] = pnt[0]*sin(pnt[1]);

        x -= xc;
        const double d = x.norm();
        //std::cout<<"["<<H/pnt[0]<<", "<<(pnt[0]/d)*sin(k*(d-pnt[0]))<<"], ";//<<std::endl;
        return (pnt[0]/d);//*exp(-T(0,1)*k*(d-pnt[0]));
    };

    auto kernel2 = [&](auto pnt) {
        const double s0=0.1;
        const double s1=sqrt(3)/3;
        const double s = 0.5 * ((s1 - s0) * pnt[0] + (s1 + s0));
        //double s = std::min(1e-4 + 0.9 * (pnt[0] + 1),1.0);
        double r =  H / s;

        const int dj=-1;
        auto tt=0.5*(pnt[1]+1);
        double theta = M_PI*pnt[1];//2*std::asin(pnt[1]);//std::exp(-1/tt)*std::exp(1);

        return kernel(Eigen::Vector2d{r,theta});
    };

    const int p = 15;
    auto interp_nodes = ChebychevInterpolation::chebnodesNd<double, p, dim>();


    Eigen::Array<T, Eigen::Dynamic, 1> interp_values(interp_nodes.cols());
    for (int i = 0; i < interp_nodes.cols(); i++) {
        const Eigen::Vector<double, dim> &pnt = interp_nodes.col(i);

        interp_values[i] = kernel2(pnt);
    }
    std::cout<<"1"<<std::endl;

    Eigen::Array<T, Eigen::Dynamic, 1> values(N);
    for (int i = 0; i < values.size(); i++) {
        values[i] = kernel2(points.col(i));
    }
    std::cout<<"2"<<std::endl;
    Eigen::Array<T, Eigen::Dynamic, 1> approx_values(N);
    //Eigen::internal::set_is_malloc_allowed(false);
    //Eigen::VectorXd interp_values(N);

    ChebychevInterpolation::parallel_evaluate<T, p, dim>(points, interp_values, approx_values);

    //Eigen::internal::set_is_malloc_allowed(true);

    approx_values -= values;

    std::cout << "error=" << approx_values.matrix().norm() / sqrt(N) << std::endl;

    return 0;

}
