#ifndef __HELMHOLTZ_IFGF_HPP__
#define __HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"


template<size_t dim >
class HelmholtzIfgfOperator : public IfgfOperator<std::complex<double>, dim,
                                                  1, HelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<double, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    HelmholtzIfgfOperator(double waveNumber,
                          size_t leafSize,
                          size_t order,
                          size_t n_elem=1,double tol=-1):
        IfgfOperator<std::complex<double>, dim, 1, HelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tol),
        k(waveNumber)
    {

    }

    typedef std::complex<double > T ;
    /*
    template<typename TX>
    inline Eigen::Vector<T, TX::ColsAtCompileTime>  kernelFunction(TX x) const
    {
	Eigen::Array<typename TX::Scalar, 1, TX::ColsAtCompileTime> d2 = x.colwise().squaredNorm();

	auto invd=(d2 < std::numeric_limits<typename TX::Scalar>::min()).select(0,Eigen::rsqrt(d2));
	
	const auto d=d2*invd;

	const double factor=1.0/ (4.0 * M_PI);        
        
        return (factor*Eigen::exp(-k * d) * invd) ;
	}*/


    inline T kernelFunction(const Eigen::Ref< const Point >&  x) const
    {
        double d = x.norm();
        return (d == 0) ? 0 : (1 / (4 * M_PI)) * exp(T(0,k) * d) / d;
    }

    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const double factor= (1.0/ (4.0 * M_PI));
	    return (Eigen::cos(k*d)+T(0,1)*Eigen::cos(k*d)) * invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=1.0/(sqrt(d2));
	    const auto d=d2*id;


	    return exp(T(0,k) * d)*id  * (1/(4.0 * M_PI));	
	}
    }

    

    template<typename TX, typename TY, typename TZ>
    inline void transfer_factor(TX x, TY xc, double H, TY pxc, double pH, TZ& result) const
    {
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> d2=(x.matrix().colwise()-xc).colwise().squaredNorm().array();
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp2=(x.matrix().colwise()-pxc).colwise().squaredNorm().array();

	/*const auto invd=Eigen::rsqrt(d2);

	const auto dp=Eigen::sqrt(dp2);
	const auto d=d2*invd;*/
	
	//double d = (x - xc).norm();
        ///Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp = (x.colwise() - pxc).norm();
	
        //result*= Eigen::exp( -k*(d-dp) )*(dp*invd);
	for (size_t i=0;i<x.cols();i++) {
	    const double d=(x.col(i).matrix()-xc).norm();
	    const double dp=(x.col(i).matrix()-pxc).norm();
	    
	    result(i)*=std::exp(T(0,k)*(d-dp))*dp/d;
	}
    }

    template<int TARGETS_AT_COMPILE_TIME>
    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, TARGETS_AT_COMPILE_TIME> >  result,IndexRange srcsIds) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

	
	for (int j = 0; j < y.cols(); j++) {
	    for (int i = 0; i < x.cols(); i++) {
	    //result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();        
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j));
	    }
        }
    }



    Eigen::Vector<T, Eigen::Dynamic>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
            const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
							     const Point& xc, double H, IndexRange srcsIds) const
    {

        Eigen::Vector<T, Eigen::Dynamic> result(y.cols());

	const int pkg_size=4;
	Eigen::Array<T, pkg_size,1> tmp;
	Eigen::Array<double, pkg_size,1> d;
        result.fill(0);        
	for (int j = 0; j < y.cols(); j++) {
            const double dc = (y.matrix().col(j) - xc).norm();

	    size_t i=0;
	    for (i = 0; i < x.cols()/ pkg_size; i++) {
		d=(x.middleCols( pkg_size*i, pkg_size).colwise()-y.col(j)).matrix().colwise().norm().array();
		tmp.real()=(Eigen::cos(k*(d-dc))*dc/d);
		tmp.imag()=Eigen::sin(k*(d-dc))*(dc/d);
		
                //const float d = (x.col(i) - y.col(j)).matrix().norm();
                result[j] +=  (weights.segment( pkg_size*i, pkg_size).matrix().transpose() * tmp.matrix()).value(); // *  std::complex<double>(  exp(std::complex<float>(0,(float) k) * (d - dc)) * (dc) / d);
	    }

	    for(size_t l=i* pkg_size;l<x.cols();l++)
	    {
		const double d = (x.col(l) - y.col(j)).matrix().norm();
                result[j] +=  weights[l] *  exp(T(0,k)* (d - dc)) * (dc) / d;
		
	    }
	}
        return result;
    }



        
    inline Eigen::Vector<int,dim> orderForBox(double H, Eigen::Vector<int,dim> baseOrder,int step=0) const
    {
	
	Eigen::Vector<int,dim> order=baseOrder;

	if(step==0) {
	    order=(baseOrder.array()-3).cwiseMax(2);
	}
	
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(double H, Eigen::Vector<int,dim> baseOrder,Eigen::Vector<size_t,dim> base, int step=0) const
    {
	const auto orders=orderForBox(H,baseOrder,step);
	Eigen::Vector<size_t,dim> els;

	if(step==0){
	    base*=3;
	    //base[2]*=2;
	}
	    
	for(int i=0;i<dim;i++) {
	    //int delta=std::ceil(std::max( std::abs(k.imag())*H/(2*(2+k.real())) , 1.0)); //make sure that k H is bounded
	    double delta=std::max( k*H/4.,1.);
	    

	    els[i]=std::max(base[i]*((int) ceil(delta)),(size_t) 1);	    
	}
	    
	return els;	    
    }



private:
    double k;

};

#endif


