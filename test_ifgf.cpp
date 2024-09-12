#include <Eigen/Dense>
#include <iostream>

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
    if(norm < 1e-14) return 0;
    /*auto kern = exp(Complex(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
	// return kern;*/

    auto kern = exp(Complex(0,kappa)*norm) / (4 * M_PI * norm);
    //x	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
	// return kern;*/
    

    return kern;
}

#include <random>
#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>


Eigen::Vector3d randomPointOnSphere() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double theta = dis(gen) * 2.0 * M_PI; // Random angle theta
    double phi = acos(2.0 * dis(gen) - 1.0); // Random angle phi

    double x = sin(phi) * cos(theta);
    double y = sin(phi) * sin(theta);
    double z = cos(phi);

    return Eigen::Vector3d(x, y, z);
}


#include <armadillo>

template <typename M>
M load_csv_arma (const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);
    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}

int main()
{
    
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray ;

    const int N = 1000;


    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
    //oneapi::tbb::task_arena arena(1);

    HelmholtzIfgfOperator<dim> op(kappa,10,8,16,-1e-8);

    PointArray srcs(3,N);
    //PointArray srcs=load_csv_arma<PointArray>("srcs.csv");

    //std::cout<<"s"<<srcs<<std::endl;
    //size_t  N=srcs.cols();
    //(dim,N);
    //srcs <<(PointArray::Random(dim,N).array());//,0.5+0.1*(PointArray::Random(dim,N).array()) ;
    for(int i=0;i<srcs.cols();i++){
	srcs.col(i)=randomPointOnSphere();
    }
    PointArray normals = srcs;//(PointArray::Random(dim,srcs.cols()).array());
    PointArray targets = srcs;//(PointArray::Random(dim, N).array());
    /*for(int i=0;i<targets.cols();i++){
	targets.col(i)=randomPointOnSphere();
	}*/


    normals.colwise().normalize();


    feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    op.init(srcs, targets);//,normals);

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> weights(srcs.cols());
    weights = Eigen::VectorXd::Random(srcs.cols());

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
