#ifndef __HELMHOLTZ_IFGF_HPP__
#define __HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"


template<size_t dim>
class HelmholtzIfgfOperator : public IfgfOperator<std::complex<double>, dim, HelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic, Eigen::RowMajor> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    HelmholtzIfgfOperator(std::complex<double> waveNumber,int leafSize, size_t order): IfgfOperator<std::complex<double>, dim, HelmholtzIfgfOperator<dim> >(leafSize,order),
										       k(waveNumber)
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
            }
        }
        return result;
    }

    inline unsigned int orderForBox(double H, unsigned int baseOrder)
    {
	
        const int order = baseOrder +  std::max(round(log(abs(imag(k)) * 2.0*H/(1.0+real(k))) / log(2)), 0.0);	
        return order;
    }

private:
    T k;

};

#endif


