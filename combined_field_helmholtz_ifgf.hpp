#ifndef __CF_HELMHOLTZ_IFGF_HPP__
#define __CF_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"
#include "util.hpp"


template<size_t dim >
class CombinedFieldHelmholtzIfgfOperator : public IfgfOperator<std::complex<double>, dim,
							     1, CombinedFieldHelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<double, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double,dim> Point;
    CombinedFieldHelmholtzIfgfOperator(double waveNumber,
				     size_t leafSize,
				     size_t order,
				     size_t n_elem=1,double tolerance=-1):
        IfgfOperator<std::complex<double>, dim, 1, CombinedFieldHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tolerance),
        kappa(waveNumber)
    {
    }


    void init(const PointArray &srcs, const PointArray targets, const PointArray& normals)
    {
	m_normals=normals;
	IfgfOperator<T,dim,1,CombinedFieldHelmholtzIfgfOperator<dim> >::init(srcs,targets);
	
    }

    //once the octree is ready, we can reorder it such that the morton-order is observed
    void onOctreeReady()
    {
	m_normals=Util::copy_with_permutation(m_normals, this->src_octree().permutation());
    }
   
    
    typedef std::complex<double > T ;

  /** CombinedFieldKernel in 3D reads
      $$ G(x-y) = \frac{1}{4\,\pi} \, \frac{e^{i\,\kappa\,|x-y|}}{|x-y|^3} \, 
          \left( \langle n_y, x-y\rangle (1- i\,\kappa\, | x-y|) - i\,\kappa\,|x-y|^2 \right), 
          \quad x, y \in \mathbb R^3, \; x\not=y\,. $$ */
    inline T  kernelFunction(const Eigen::Ref< const Point >&  x,const Eigen::Ref< const Point >&  n) const
    {
	double norm = x.norm();

	T nxy = -n.dot(x);

	auto kern = exp(T(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
	    * ( nxy * (std::complex<double>(1.,0)*T(1.) - std::complex<double>(0,kappa)*norm)  - T(0,kappa)*norm*norm);
	
	return norm<1e-14 ? 0.0 : kern;
	    
    }

    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const double factor= (1.0/ (4.0 * M_PI));
	    return Eigen::exp(std::complex(0.,kappa) * d) * invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=1.0/(sqrt(d2));
	    const auto d=d2*id;

	    return exp(std::complex(0.,kappa) * d)*id  * (1/(4.0 * M_PI));	    
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
	      
	result*=(Eigen::exp( std::complex(0.,kappa)*(d-dp) )*(dp*invd));
	
    }


    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, Eigen::Dynamic> >  result,IndexRange srcIds) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

	for (int i = 0; i < x.cols(); i++) {
	    //result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
	    for (int j = 0; j < y.cols(); j++) {
		result[j] += w[i] * kernelFunction(x.col(i) - y.col(j),m_normals.col(srcIds.first+i));
	    }
	}	
    }

    Eigen::Array<T, Eigen::Dynamic,1>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
							      const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
							      const Point& xc, double H,IndexRange srcIds) const
    {
	Eigen::Array<T, Eigen::Dynamic,1> result(y.cols());


	const std::complex<double> k=-std::complex<double>(0,kappa);
        result.fill(0);
	for (int j = 0; j < y.cols(); j++) {
	    const double dc = (y.matrix().col(j) - xc).norm();
	    for (int i = 0; i < x.cols(); i++) {
		const Point& z=y.matrix().col(j)-x.matrix().col(i);

                const auto d2 = z.squaredNorm();
		const auto nxy=z.dot(m_normals.col(srcIds.first+i).matrix());
		
		const auto id= (d2>1e-12) ? 1.0/sqrt(d2) : 0;
		const auto d=d2*id;
		
		//if(d>1e-12) {
		//const double w= (-(1.0/(d2)) *dc*  (x.col(i)-y.col(j)).matrix().dot(m_normals.col(srcIds.first+i).matrix()));
		//result.row(j) +=  weights[i] * exp(-k * (  d-dc)) *(-k-id) * w;
		//result.row(j) += -std::complex<double>(0.,kappa)*(   (d==0) ? 0 : weights[i] *    exp(-k * (d - dc)) * (dc) / d);

		//	auto kern = exp(T(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
		//* ( nxy * (std::complex<double>(1.,0)*T(1.) - std::complex<double>(0,kappa)*norm)  - T(0,kappa)*norm*norm);


                result.row(j) += weights[i]* ( exp(T(0, kappa) * (d - dc)) * dc / (d*d*d)
					       * ( nxy * (std::complex<double>(1.,0)*T(1.) - std::complex<double>(0,kappa)*d)  - T(0,kappa)*d*d));

		
		//exp(-k * (d - dc))*dc * (-1.0 /(d*d))*(k+1.0/d)*(x(dx,i)-y(dx,j));
		//}
	    }
	}
        return result;        
    }
 
    inline Eigen::Vector<int,dim> orderForBox(double H, unsigned int baseOrder) const
    {
	
	Eigen::Vector<int,dim> order;
	order.fill(baseOrder);
	order[0]=std::max((int)  baseOrder-2,1);
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(double H, unsigned int baseOrder,Eigen::Vector<size_t,dim> base) const
    {
	const unsigned int order=orderForBox(H,baseOrder).minCoeff();
	double delta=std::max( kappa*H/(order) , 1.0); //make sure that k H/p is bounded. this guarantees spectral convergence w.r.t. p.
	base*=(int) ceil(delta);
	return base;	    
    }


private:
    double kappa;
    PointArray m_normals;
};

#endif
