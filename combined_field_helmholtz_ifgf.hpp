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
	const double d2 = x.squaredNorm();
        const double invd = d2<1e-14 ?  0.0 : (1.0/sqrt(d2));
        const double d=d2*invd;
        

        double nxy = -n.dot(x);
        
        const double s=sin(kappa*(d));
        const double c=cos(kappa*(d));


        
        const double f=(1.0/(4.*M_PI))*invd*invd*invd;//*d2*d);
                

        double real=f*((c)*nxy + (s)*kappa*(d*nxy+d2));
        double imag=f*(-(c)*kappa*(d*nxy+d2) + (s)*nxy);

        return T(real,imag);

	/*double nxy = -n.dot(x);
	auto kern = exp(T(0,kappa)*d) / (4 * M_PI * d2*d)
	    * ( nxy * (std::complex<double>(1.,0)*T(1.) - std::complex<double>(0,kappa)*d)  - T(0,kappa)*d2);
	
            return d<1e-14 ? 0.0 : kern;*/
	    
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

    
    template<int TARGETS_AT_COMPILE_TIME>
    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Array<T, TARGETS_AT_COMPILE_TIME,1> >  result,IndexRange srcIds) const
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


        const auto w_r=weights.real().eval();
        const auto w_i=weights.imag().eval();
            
        result.fill(0);
	for (int j = 0; j < y.cols(); j++) {
            const double dc = (y.matrix().col(j) - xc).norm();


            for (int i = 0; i < x.cols(); i++) {
                
                const Point& z=y.matrix().col(j)-x.matrix().col(i);

                const double d2 = z.squaredNorm();
		if(d2>1e-14 ){

		    const double nxy=z.dot(m_normals.col(srcIds.first+i).matrix());
                
		    const double id= 1.0/sqrt(d2);
		    const double d=d2*id;
                
		    const double s=sin(kappa*(d-dc));
		    const double c=cos(kappa*(d-dc));


		    const double f=dc/(d2*d);


		    result.row(j).real()+=f*((w_r[i]*c-w_i[i]*s)*nxy + (w_r[i]*s+w_i[i]*c)*kappa*(d*nxy+d2));
		    result.row(j).imag()+=f*(-(w_r[i]*c-w_i[i]*s)*kappa*(d*nxy+d2) + (w_r[i]*s+w_i[i]*c)*nxy);
		}

                
                
                //result.row(j) += weights[i]* ( exp(T(0, kappa) * (d - dc)) * dc / (d2*d)
                //                              * ( nxy * (std::complex<double>(1.,0)*T(1.)
                //                              - std::complex<double>(0,kappa)*d)  - T(0,kappa)*d*d));
            }
        
        }
        
        return result;        
    }
 

    
    inline Eigen::Vector<int,dim> orderForBox(double H, unsigned int baseOrder,int step=0) const
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
	const auto orders=orderForBox(H,baseOrder,0);
	Eigen::Vector<size_t,dim> els;
	if(step==1) {
	    base[0]=1;
	    base[1]=2;
	    base[2]=4;	    
	}
	
	for(int i=0;i<dim;i++) {
	    double delta=std::max( kappa*H/(orders[i]) , 1.0); //make sure that k H/p is bounded by 1. this guarantees spectral convergence w.r.t. p.	   
	    els[i]=base[i]*((int) ceil(delta));	    
	}

	return els;	    
    }



private:
    double kappa;
    PointArray m_normals;
};

#endif
