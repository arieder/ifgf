#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "ifgfoperator.hpp"
#include "octree.hpp"

const int dim=2;
typedef Eigen::Vector<double,dim> Point;

const std::complex<double>  k = std::complex<double>(0, 4);

std::complex<double> kernel(const Point& x, const Point& y)
{
    double d = (x - y).norm();

    return d == 0 ? 0 : (1 / (4 * M_PI)) * exp(-k * d) / d;
}

class MyIfgfOperator : public IfgfOperator<std::complex<double>, dim, MyIfgfOperator>
{
public:
    MyIfgfOperator(int leafSize, size_t order): IfgfOperator(leafSize,order)
    {
    }

    typedef std::complex<double > T ;

    template<typename TX>
    inline T kernelFunction(TX x) const
    {
        double d = x.norm();
        return (d == 0) ? 0 : (1 / (4 * M_PI)) * exp(-k * d) / d;
    }

    template<typename TX>
    inline T CF(TX x) const
    {
        double d = x.norm();
        return exp(-k * d) / (4 * M_PI * d);
    }

    template<typename TX, typename TY>
    inline T transfer_factor(TX x, TY xc, double H, TY pxc, double pH) const
    {
        /*yc  = IFGF.center(Y)
          yp  = IFGF.center(IFGF.parent(Y))
          d   = norm(x-yc)
          dp  = norm(x-yp)
          exp(im*K.k*(d-dp))*dp/d
        */
        double d = (x - xc).norm();
        double dp = (x - pxc).norm();

	
        return d==0 ? 1 : exp(-k * (d - dp)) * (dp / d);
    }

    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic> >  result) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

        for (int i = 0; i < x.cols(); i++) {
            for (int j = 0; j < y.cols(); j++) {
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j));
            }
        }
    }

    Eigen::Vector<T, Eigen::Dynamic>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
            const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
            const Point& xc, double H) const
    {

        Eigen::Vector<T, Eigen::Dynamic> result(y.cols());
        result.fill(0);
        for (int j = 0; j < y.cols(); j++) {
            double dc = (y.col(j) - xc).norm();
            for (int i = 0; i < x.cols(); i++) {
                double d = (x.col(i) - y.col(j)).norm();
		if(!std::isfinite(d))
		    continue;
                result[j] +=
		    (d==0) ? 0 : weights[i] * 
		    exp(-k * (d - dc)) * (dc) / d;
                //kernelFunction(x.col(i)-y.col(j))*inv_CF(y.col(j)-xc);
            }
        }
        return result;
    }

    inline unsigned int orderForBox(double H, unsigned int baseOrder)
    {
	
        const int order = baseOrder ;//+  std::max(round(log(abs(k) * 2*H) / log(2)), 0.0);	
        return order;
    }

};

#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>
int main()
{
    const int N = 10000;
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray ;


    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
    //oneapi::tbb::task_arena arena(1);

    MyIfgfOperator op(100,25);

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
