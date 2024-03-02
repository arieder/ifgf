#ifndef __GRAD_HELMHOLTZ_IFGF_HPP__
#define __GRAD_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"


template<size_t dim >
class GradHelmholtzIfgfOperator : public IfgfOperator<std::complex<double>, dim,
                                                 dim, GradHelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    GradHelmholtzIfgfOperator(std::complex<double> waveNumber,
                          size_t leafSize,
                          size_t order,
                          size_t n_elem=1):
        IfgfOperator<std::complex<double>, dim, 1, HelmholtzIfgfOperator<dim> >(leafSize,order, n_elem),
        k(waveNumber)
    {
    }

    typedef std::complex<double > T ;

    template<typename TX>
    inline Eigen::Array<T, dim, TX::ColsAtCompileTime>  kernelFunction(TX x) const
    {
	Eigen::Array<typename TX::Scalar, 1, TX::ColsAtCompileTime> d2 = x.colwise().squaredNorm();

	auto invd=(d2 < std::numeric_limits<typename TX::Scalar>::min()).select(0,Eigen::rsqrt(d2));
	
	const auto d=d2*invd;

	const double factor=1.0/ (4.0 * M_PI);        

        return factor*
            (d2 > 1e-12).select( ((-1.0 / d2) * Eigen::exp(- k * d).colwise() * (k + invd).colwise() * x) ,
                                 Eigen::Vector<T, dim>::Zero()).transpose();
    }



    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const double factor= (1.0/ (4.0 * M_PI));
	    return Eigen::exp(-k * d) * invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=1.0/(sqrt(d2));
	    const auto d=d2*id;


	    return exp(-k * d)*id  * (1/(4.0 * M_PI));	
	}
    }


    

    template<typename TX, typename TY, typename TZ>
    inline void transfer_factor(TX x, TY xc, double H, TY pxc, double pH, TZ& result) const
    {
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> d2=(x.colwise()-xc).colwise().squaredNorm().array();
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp2=(x.colwise()-pxc).colwise().squaredNorm().array();

	const auto invd=Eigen::rsqrt(d2);

	const auto dp=Eigen::sqrt(dp2);
	const auto d=d2*invd;
	
	//double d = (x - xc).norm();
        ///Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp = (x.colwise() - pxc).norm();
	
        result*= Eigen::exp( -k*(d-dp) )*(dp*invd);
	/*for (size_t i=0;i<x.cols();i++) {
	    const double d=(x-xc).norm();
	    const double dp=(x-pxc).norm();
	    
	    result(i)*=std::exp(-k*(d-dp))*dp/d;
	    }*/
    }

    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic> >  result) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

        for (int i = 0; i < x.cols(); i++) {
	    result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
            /*for (int j = 0; j < y.cols(); j++) {
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j));
		}*/
        }
    }

    Eigen::Array<T, Eigen::Dynamic,dim>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
                                                                const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
            const Point& xc, double H) const
    {

        Eigen::Array<T, Eigen::Dynamic,dim> result(y.cols());


        result.fill(0);        
	for (int j = 0; j < y.cols(); j++) {
            const double dc = (y.col(j) - xc).norm();

	    for (int i = 0; i < x.cols(); i++) {
                double d = (x.col(i) - y.col(j)).norm();
		
                result.row(j) +=
		    (d==0) ? Eigen::Vector<T,dim>::Zeros() : weights[i] * 
		    exp(-k * (d - dc))*dc * (-1.0 /(d*d))*(k+1.0/d)*(x.col(i)-y.col(j)).transpose();
	    }
        }
        return result;
    }

    inline unsigned int orderForBox(double H, unsigned int baseOrder) const
    {
	
        const int order = baseOrder; // +  std::max(round(H*imag(k)), 0.0);	
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(double H, unsigned int baseOrder,Eigen::Vector<size_t,dim> base) const
    {
	const unsigned int order=orderForBox(H,baseOrder);
	double delta=std::max( 2*abs(imag(k))*H/(order*(1.0+real(k))) , 1.0); //make sure that k H/p is bounded. this guarantees spectral convergence w.r.t. p.
	base*=(int) ceil(delta);
	return base;	    
    }


private:
    std::complex<double> k;

};

#endif
