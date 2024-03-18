#ifndef __GRAD_HELMHOLTZ_IFGF_HPP__
#define __GRAD_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"


template<size_t dim >
class GradHelmholtzIfgfOperator : public IfgfOperator<std::complex<double>, dim,
						      1, GradHelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<double, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    GradHelmholtzIfgfOperator(std::complex<double> waveNumber,
                          size_t leafSize,
                          size_t order,
			      size_t n_elem=1,double tolerance=-1):
        IfgfOperator<std::complex<double>, dim, 1, GradHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tolerance),
        k(waveNumber)
    {
    }

    typedef std::complex<double > T ;

    /*template<typename TX>
    inline Eigen::Array<T, dim, TX::ColsAtCompileTime>  kernelFunction(TX x) const
    {
    Eigen::Array<typename TX::Scalar, 1, TX::ColsAtCompileTime> d2 = x.colwise().squaredNorm();

	auto invd=(d2 < std::numeric_limits<typename TX::Scalar>::min()).select(0,Eigen::rsqrt(d2));
	
	const auto d=d2*invd;

	const T factor=1.0/ (4.0 * M_PI);        

	Eigen::Array<T,dim, TX::ColsAtCompileTime> res= x.colwise()*(factor*((-1.0 / d2) * Eigen::exp(- k * d) * (k + invd))).array() ;
	return res.array();
	//(d2 > 1e-12).select( ((-1.0 / d2) * Eigen::exp(- k * d).colwise() * (k + invd).colwise() * x) ,
        //                         Eigen::Array<T, TX::ColsAtCompileTime,dim>::Zero(x.cols(),dim)).transpose();
	}*/

    template <int dx>
    inline T  kernelFunction(const Eigen::Ref< const Point >&  x) const
    {
	double d = x.norm();

	if constexpr(dx==-1) {
	    double d = x.norm();
	    return (d == 0) ? 0 : (1 / (4 * M_PI)) * exp(-k * d) / d;
	}else{
	    return d<1e-12 ? 0.0:   -(1.0 / (4.0 * M_PI)) * (1.0/(d*d)) * exp(-k * d) *(-k-1.0/d)*x[m_dx];
	}
    }

    void setDx(int dx) {
	m_dx=dx;
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
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> d2=(x.matrix().colwise()-xc).colwise().squaredNorm().array();
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp2=(x.matrix().colwise()-pxc).colwise().squaredNorm().array();

	const auto invd=Eigen::rsqrt(d2);

	const auto dp=Eigen::sqrt(dp2);
	const auto d=d2*invd;
	
	//double d = (x - xc).norm();
        ///Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp = (x.colwise() - pxc).norm();



	result*=(Eigen::exp( -k*(d-dp) )*(dp*invd));
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

	switch(m_dx) {
	case -1:
	    for (int i = 0; i < x.cols(); i++) {
		//result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
		for (int j = 0; j < y.cols(); j++) {
		    result[j] += w[i] * kernelFunction<-1>(x.col(i) - y.col(j));
		}
	    }
	    break;
	case 0:
	    for (int i = 0; i < x.cols(); i++) {
		//result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
		for (int j = 0; j < y.cols(); j++) {
		    result[j] += w[i] * kernelFunction<0>(x.col(i) - y.col(j));
		}
	    }
	    break;
	case 1:
	    for (int i = 0; i < x.cols(); i++) {
		//result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
		for (int j = 0; j < y.cols(); j++) {
		    result[j] += w[i] * kernelFunction<1>(x.col(i) - y.col(j));
		}
	    }
	    break;
	case 2:
	    for (int i = 0; i < x.cols(); i++) {
		//result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
		for (int j = 0; j < y.cols(); j++) {
		    result[j] += w[i] * kernelFunction<2>(x.col(i) - y.col(j));
		}
	    }    
	}
    }

    Eigen::Array<T, Eigen::Dynamic,1>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
                                                                const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
            const Point& xc, double H) const
    {

        Eigen::Array<T, Eigen::Dynamic,1> result(y.cols());


        result.fill(0);
	if(m_dx==-1) {
	    for (int j = 0; j < y.cols(); j++) {
		const double dc = (y.matrix().col(j) - xc).norm();

		for (int i = 0; i < x.cols(); i++) {
		    const double d2 = (x.col(i) - y.col(j)).matrix().squaredNorm();
		    const double id=1.0/sqrt(d2);
		    const double d=d2*id;

		    result[j] +=
			(d==0) ? 0 : weights[i] * 
			exp(-k * (d - dc)) * (dc) *id;
		}
	    }
	}else{
	    for (int j = 0; j < y.cols(); j++) {
		const double dc = (y.matrix().col(j) - xc).norm();
                for (int i = 0; i < x.cols(); i++) {
		    const double d2 = (x.matrix().col(i) - y.matrix().col(j)).squaredNorm();

		    const double id=1.0/sqrt(d2);
		    const double d=d2*id;

		    if(d>1e-12) {
			result.row(j) +=  weights[i] *
			    -(1.0/(d2)) *dc* exp(-k * (d-dc)) *(-k-id)*(x(m_dx,i)-y(m_dx,j));
			//exp(-k * (d - dc))*dc * (-1.0 /(d*d))*(k+1.0/d)*(x(dx,i)-y(dx,j));
		    }
		}
	
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
	double delta=std::max( abs(imag(k))*H/(order*(1.0+real(k))) , 1.0); //make sure that k H/p is bounded. this guarantees spectral convergence w.r.t. p.
	base*=(int) ceil(delta);
	return base;	    
    }


private:
    std::complex<double> k;
    int m_dx;
};

#endif
