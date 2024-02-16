#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include "chebinterp.hpp"
#include <complex>

#include <tbb/parallel_for.h>

int main()
{
    const int N=500;
    const int dim=3;
    const int POINTS_AT_COMPILE_TIME=16;
    typedef std::complex<double> T;
    Eigen::Array<double,dim,Eigen::Dynamic,Eigen::RowMajor> points=Eigen::Array<double,dim,Eigen::Dynamic,Eigen::RowMajor>::Random(dim,N);


    
    auto kernel= [](auto pnt)
    {
	const double d=1.0*pnt[0]+2.0*pnt[1]+2.0*pnt[2];//pnt.sum();//(pnt.array()+1.0).matrix().norm();
	return std::complex(d*exp(std::complex<double>(2,1)*d)/d);
    };
    

    const int p=50;
    

    auto interp_nodes=ChebychevInterpolation::chebnodesNd<double,p,dim>();
    Eigen::Array<T,Eigen::Dynamic,1> interp_values(p*p*p);
    
    for(int i=0;i<interp_nodes.cols();i++)
    {
	const Eigen::Vector<double,dim>& pnt=interp_nodes.col(i);

	
	interp_values[i]=kernel(pnt);
    }

    Eigen::Array<T,Eigen::Dynamic,1> values(N);
    for(int i=0;i<values.size();i++)
    {
	values[i]=kernel(points.col(i));
    }

    Eigen::Array<T,Eigen::Dynamic,1> approx_values(N);
    //Eigen::internal::set_is_malloc_allowed(false);
    //Eigen::VectorXd interp_values(N);
    
    ChebychevInterpolation::parallel_evaluate<T,p,dim>(points,interp_values,approx_values);



    //Eigen::internal::set_is_malloc_allowed(true);
    
    approx_values-=values;

    std::cout<<"error="<<approx_values.matrix().norm()/sqrt(N)<<std::endl;

    return 0;

    
}
