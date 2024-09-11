#ifndef __LAPLACE_IFGF_HPP__
#define __LAPLACE_IFGF_HPP__

#include "ifgfoperator.hpp"


template<size_t dim >
class LaplaceIfgfOperator : public IfgfOperator<double, dim,
                                                  1, LaplaceIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<double, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    LaplaceIfgfOperator(  size_t leafSize,
                          size_t order,
                          size_t n_elem=1,double tol=-1):
        IfgfOperator<double, dim, 1, LaplaceIfgfOperator<dim> >(leafSize,order, n_elem,tol)
    {
    }

    typedef double T ;
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
        return (d == 0) ? 0 : (1 / (4 * M_PI) ) / d;
    }

    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const double factor= (1.0/ (4.0 * M_PI));
	    return invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=1.0/(sqrt(d2));

	    return id  * (1/(4.0 * M_PI));	
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
	    
	    result(i)*=dp/d;
	}
    }

    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic> >  result,IndexRange srcsIds) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

        for (int i = 0; i < x.cols(); i++) {
	    //result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
            for (int j = 0; j < y.cols(); j++) {
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j));
	    }
        }
    }

    Eigen::Vector<T, Eigen::Dynamic>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
            const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
							     const Point& xc, double H, IndexRange srcsIds) const
    {

        Eigen::Vector<T, Eigen::Dynamic> result(y.cols());

        result.fill(0);        
	for (int j = 0; j < y.cols(); j++) {
            const double dc = (y.matrix().col(j) - xc).norm();

	    for (int i = 0; i < x.cols(); i++) {
                double d = (x.col(i) - y.col(j)).matrix().norm();
		
                result[j] +=
		    (d==0) ? 0 : weights[i] * 
		    (dc) / d;
	    }
        }
        return result;
    }

    inline Eigen::Vector<int,dim> orderForBox(double H, unsigned int baseOrder, int step=0) const
    {
	
	Eigen::Vector<int,dim> order;
	order.fill(baseOrder);
	order[0]=std::max((int) baseOrder-2,1);
	
	if(step==1) {
	    for(int i=0;i<dim;i++){
		order[i]=(int) 2*order[i];
	    }
	}

	
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(double H, unsigned int baseOrder,Eigen::Vector<size_t,dim> base, int step=0) const
    {
	const unsigned int order=orderForBox(H,baseOrder).minCoeff();

	if(step==1) {
	    base[0]=1;
	    base[1]=2;
	    base[2]=4;	    
	}

	return base;	    
    }



};

#endif


