#ifndef _CHEBINTERP_HPP_
#define _CHEBINTERP_HPP_

#include "Eigen/src/Core/util/Constants.h"
#include "boundingbox.hpp"
#include <cstddef>
#include <tuple>

#include <fenv.h>
#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <exception>
#include <iostream>

#include <tbb/spin_mutex.h>

#include "cone_domain.hpp"

#ifdef USE_NGSOLVE
    #include </home/arieder/devel/install/include/core/ngcore.hpp>
#endif

namespace ChebychevInterpolation
{

    template< typename T, int N_AT_COMPILE_TIME>
    inline constexpr Eigen::Array<T,N_AT_COMPILE_TIME,1> chebnodes1d(int n)
    {
	std::cout<<"new chebnodes"<<n<<std::endl;
        assert(N_AT_COMPILE_TIME==-1 || n==N_AT_COMPILE_TIME);
        //return (Eigen::Array<T, N_AT_COMPILE_TIME, 1>::LinSpaced((int) n, 0, M_PI)).cos();
	Eigen::Array<T, N_AT_COMPILE_TIME, 1> nodes(n);
	 
	for(int i=0;i<n;i++) {
	    nodes[i]=cos((2.*i+1.0)/(2.*n)*M_PI);
	}
	return nodes;
    }


    template< typename T>
    Eigen::Array<T,Eigen::Dynamic,1> cachedChebnodes1d(int n)
    {
	static std::unordered_map<size_t, Eigen::Array<T, Eigen::Dynamic,1 > >  cache;
	if (cache.count(n) > 0) {
	    return cache[n];
	} else {	    
	    cache[n] = chebnodes1d<T, Eigen::Dynamic>(n);
	    return cache[n];
	}
    }


    template <typename T, int N_AT_COMPILE_TIME>
    inline Eigen::Array<T, 1, Eigen::Dynamic>
    chebnodesNd(const Eigen::Ref<const Eigen::Vector<int, 1>> &ns) {
        const Eigen::Array<T, N_AT_COMPILE_TIME, 1> nodes1d =
            chebnodes1d<T, N_AT_COMPILE_TIME>(ns[0]);
        Eigen::Array<T, 1, Eigen::Dynamic> nodesNd = nodes1d.transpose();
        return nodesNd;
    }

    template <typename T, int NR_AT_COMPILE_TIME, int NTHETA_AT_COMPILE_TIME>
    inline Eigen::Array<T, 2, Eigen::Dynamic>
    chebnodesNd(const Eigen::Ref<const Eigen::Vector<int, 2> > &ns) {
        const Eigen::Array<T, NR_AT_COMPILE_TIME, 1> nodesR =
            chebnodes1d<T, NR_AT_COMPILE_TIME>(ns[0]);
        const Eigen::Array<T, NTHETA_AT_COMPILE_TIME, 1> nodesTheta =
            chebnodes1d<T, NTHETA_AT_COMPILE_TIME>(ns[1]);

        Eigen::Array<T, 2, Eigen::Dynamic> nodesNd(2, nodesR.size() *
						   nodesTheta.size());

        for (size_t i = 0; i < nodesTheta.size(); ++i) {
          for (size_t j = 0; j < nodesR.size(); j++) {
	      nodesNd(0, i * nodesR.size() + j) = nodesR[j];
	      nodesNd(1, i * nodesR.size() + j) = nodesTheta[i];
          }
        }
        return nodesNd;
    }

    template <typename T, int NR_AT_COMPILE_TIME, int NTHETA_AT_COMPILE_TIME,
              int NPHI_AT_COMPILE_TIME>
    inline Eigen::Array<T, 3, Eigen::Dynamic>
    chebnodesNd(const Eigen::Vector<int, 3> &ns) {
        const Eigen::Array<T, NR_AT_COMPILE_TIME, 1> nodesR =
            chebnodes1d<T, NR_AT_COMPILE_TIME>(ns[0]);
        const Eigen::Array<T, NTHETA_AT_COMPILE_TIME, 1> nodesTheta =
            chebnodes1d<T, NTHETA_AT_COMPILE_TIME>(ns[1]);
        const Eigen::Array<T, NPHI_AT_COMPILE_TIME, 1> nodesPhi =
            chebnodes1d<T, NPHI_AT_COMPILE_TIME>(ns[2]);

        Eigen::Array<T, 3, Eigen::Dynamic> nodesNd(
            3, nodesR.size() * nodesTheta.size() * nodesPhi.size());


        const int nR = nodesR.size();
        const int nP = nodesPhi.size();
        const int nT = nodesTheta.size();

        for (size_t i = 0; i < nP; ++i) {
          for (size_t j = 0; j < nT; j++) {
            for (size_t k = 0; k < nR; k++) {
              nodesNd(0, i * nT * nR + j * nR + k) = nodesR[k];
              nodesNd(1, i * nT * nR + j * nR + k) = nodesTheta[j];
              nodesNd(2, i * nT * nR + j * nR + k) = nodesPhi[i];
            }
          }
        }
        return nodesNd;
    }

    template <typename T>
    inline const Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic> &chebvals( size_t n)
    {
	static tbb::spin_mutex mutex;
	
        static std::unordered_map<size_t, Eigen::Array<T, Eigen::Dynamic,Eigen::Dynamic> >
            cache;

	const size_t key=n;
	if (cache.count(key) > 0) {
	    return cache[key];
	} else {
	    tbb::spin_mutex::scoped_lock lock(mutex);
	    /*	    if (cache.count(key) > 0) {
		return cache[key];
		}*/


	    Eigen::Array<T, Eigen::Dynamic,Eigen::Dynamic> arr(n,n);
	    for(size_t idx=0;idx<n;idx++) {
		for(size_t sigma=0;sigma<n;sigma++) {
		    const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*n));
		    arr(sigma,idx)=Td;
		}
	    }
		
	    cache[key] =  arr;
	    
	    return cache[key];
	}        
    }



    template<typename TV,typename TV2>
    auto   computeDiffVector(int n, const TV& x,const TV2& nodes)
    {	
	Eigen::Array<double, -1, 1 > c(n,1);
	for(size_t i=0;i<n;i++){
	    c[i]=(i % 2 == 0) ? 1.0:-1.0;
	}
	c[0]*=0.5;
	c[n-1]*=0.5;

	Eigen::Array<typename TV::Scalar, Eigen::Dynamic, Eigen::Dynamic> diff(x.size(),n);
	for(long int i=0;i<x.size();i++) {
	    diff.row(i)=c/(x[i]-nodes.array()+1e-12);
	}
	return diff;
    }

    template<typename TV, typename  TP>
    auto inline  transformNodesToInterval(TV& nodes, TP min, TP max)
    {
	const TP a=(max-min)/2.0;
	const TP b=(max+min)/2.0;
	nodes=a*nodes+b;
    }

    
    
    template <typename T,int DIM>
    void chebtransform(const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1> >& src,
			      Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > dest,
			      const Eigen::Ref<const Eigen::Vector<int,DIM> >& ns
			      )
    {
	//evaluate all line-functions
	int nsigma=ns.head(DIM-1).prod();
	if constexpr(DIM==1) {
	    nsigma=1;
	}
	const int stride=nsigma;
	dest.fill(0);

	const auto& cv=chebvals<double>( ns[DIM-1]);

	//std::cout<<"ns="<<ns<<std::endl;
	//std::cout<<"nsigma="<<nsigma<<std::endl;
	if constexpr(DIM==1) {
	    //do a straight forwad summation of the innermost dimension
	    
	    for(size_t idx=0;idx<ns[0];idx++) {
		for(size_t sigma=0;sigma<ns[0];sigma++)  {
		    const double Td=cv(sigma,idx);//cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[0]));
		    dest.segment(idx*stride,nsigma)+=src.segment(sigma*stride,nsigma)*Td;
		}
		dest.segment(idx*stride,nsigma)*=(idx ==0 ? 1.:2. )*1./ns[0];
	    }
	    //std::cout<<"done inner"<<std::endl;
	}else {
#if 0
	    std::cout<<"slow for testing"<<std::endl;
	    Eigen::Array<T, Eigen::Dynamic,1> D(dest.rows());
	    D.fill(0);
	    for(size_t idx=0;idx<ns[0];idx++) {
		for(size_t idx2=0;idx2<ns[1];idx2++) {		
		    for(size_t sigma=0;sigma<ns[0];sigma++)  {
			for(size_t sigma2=0;sigma2<ns[1];sigma2++)  {
			    const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[0]));
			    const double Td2=cos(idx2*M_PI*(2.*sigma2+1.)/(2.*ns[0]));
			    D(idx*ns[1]+ idx2)+=src(sigma*ns[1]+sigma2)*Td*Td2 * ((idx ==0 ? 1.:2. )*1./ns[0])*((idx2 ==0 ? 1.:2. )*1./ns[1]);
			}
		    }		    
		}
	    }
#endif
	    Eigen::Array<T, Eigen::Dynamic, 1> M(ns.prod());
	    //std::cout<<"building m"<<DIM<<std::endl;
	    for(size_t idx=0;idx<ns[DIM-1];idx++) {
		//std::cout<<"idx"<<idx<<" "<<idx*stride<<" "<<M.size()<<" "<<src.size()<<" "<<nsigma<<std::endl;
		chebtransform<T,DIM-1>(src.segment(idx*stride,nsigma),M.segment(idx*stride,nsigma),ns.template head<DIM-1>());
	    }

	    for(size_t idx=0;idx<ns[DIM-1];idx++) {
		for(size_t sigma=0;sigma<ns[DIM-1];sigma++)  {
		    const double Td=cv(sigma,idx);//cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[0]));
		    //const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));
		    dest.segment(idx*stride,nsigma)+=M.segment(sigma*stride,nsigma)*Td;
		}	    
		dest.segment(idx*stride,nsigma)*=(idx ==0 ? 1:2 )*1./ns[DIM-1];
	    }

	    //std::cout<<"err sum_factor="<<(D-dest).matrix().norm()<<std::endl;
	}
    }

    template <typename T, int POINTS_AT_COMPILE_TIME, int DIM, int DIM_X, unsigned int DIMOUT, int ND=-1, int... Ns>    
    class ClenshawEvaluator
    {
    public:
	inline  Eigen::Array<T,POINTS_AT_COMPILE_TIME,DIMOUT>
	operator()(const Eigen::Ref<const Eigen::Array<double,DIM_X, POINTS_AT_COMPILE_TIME> > &x,
		   const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> > &vals,
		   const Eigen::Ref<const Eigen::Vector<int,DIM_X> >& ns )
    {
	static_assert(DIMOUT==1); //For now only 1d output works.
	static_assert(DIM>0);
	static_assert(DIM<=DIM_X);
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> b1(x.cols(),DIMOUT);
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> b2(x.cols(),DIMOUT);
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> tmp(x.cols(),DIMOUT);

	const int Nd= ND >0 ? ND : ns[DIM-1];

	/*	std::cout<<"ND="<<ND<<" vs "<<ns<<" "<<DIM_X<<" \n";
	(std::cout<<...<<Ns);
	std::cout<<"\n";*/
	    
	    
	//assert(ND < 0 || ND==ns[DIM-1]);
	
	if constexpr (DIM<=1)	    
	{
	    const int Nd= ND >0 ? ND : ns[0];
	    if(Nd<=2) {
		if(Nd==1) {
		    return vals[0]+ x.row(0).transpose().Zero();
		}else {
		    return x.row(0).transpose()*vals[1]+vals[0];			
		}
	    }
	    b1=2.*x.row(0)*vals(Nd-1)+vals(Nd-2);
	    b2.fill(vals(Nd-1));
	

	    for(size_t j=Nd-3;j>0;j--) {
		tmp=(2.*((b1)*x.row(0).transpose())-(b2))+vals(j);
	    
		b2=b1;
		b1=tmp;
	    
	    }
	    
	    return (b1*x.row(0).transpose()-b2)+vals(0);
	}else //recurse down
	{	   	    
	    const size_t stride = ns.template head<DIM-1>().prod();

	    ClenshawEvaluator<T, POINTS_AT_COMPILE_TIME, std::max(DIM-1,1), DIM_X,DIMOUT,Ns...> clenshaw;
	    if(Nd<=2) {
		const Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT>& c0=clenshaw(x,
										   vals.middleRows(0 * stride, stride),
										   ns).eval();

		if(Nd==1) {
		    return c0 + x.row(DIM-1).transpose().Zero();
		}else {
		    b1=clenshaw(x,
				vals.middleRows((1) * stride, stride),
				ns).eval();

		    return x.row(DIM-1).transpose()*b1+c0;		    
		}
	    }


	    
	    b2=clenshaw(x,
			vals.middleRows((Nd-1) * stride, stride),
			ns).eval();
	    const Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> cn2=clenshaw(x,
			      vals.middleRows((Nd-2) * stride, stride),
			      ns).eval();


	    b1=2.*b2*x.row(DIM-1).transpose()+cn2;

	    const Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> c0=clenshaw(x,
			     vals.middleRows(0 * stride, stride),
			     ns).eval();
	    for(size_t j=Nd-3;j>0;j--) {
		tmp= clenshaw(x,
			      vals.middleRows(j * stride, stride),
			      ns);
		tmp+=(2.*(b1*x.template row(DIM-1).transpose())-b2);
		b2=b1;
		b1=tmp;
		
	    }

	    return (b1*x.row(DIM-1).transpose()-b2) + c0;

	}
    }
    };




    //3d evaluation code
    template <typename T, typename PointScalarType, int DIMOUT>
    inline Eigen::Array<T, Eigen::Dynamic, DIMOUT>
    evaluate(
	     const Eigen::Ref<const Eigen::Array<PointScalarType , 3, Eigen::Dynamic> > &x,
             const Eigen::Ref<const Eigen::Array<T , Eigen::Dynamic, DIMOUT> > &vals,
             const Eigen::Vector<int, 3>& ns,
             BoundingBox<3> box = BoundingBox<3>()  )
    {
	const int DIM=3;



#ifdef USE_NGSOLVE	
      static ngcore::Timer t("ngbem ifgf cheb::eval");
      static ngcore::Timer tmult("ngbem ifgf cheb::eval mult");
      static ngcore::Timer tmultx("ngbem ifgf cheb::eval multx");
      static ngcore::Timer tmultyz("ngbem ifgf cheb::eval multyz");                  
      ngcore::RegionTimer reg(t);

      t.AddFlops (ns[0]*ns[1]*ns[2]*x.cols()*DIMOUT);
#endif

        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(x.cols(), DIMOUT);
        result.fill(0);

	

        assert(DIM == x.rows());
        Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes1 =
            cachedChebnodes1d<PointScalarType>(ns[0]);        
        Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes2 =
            cachedChebnodes1d<PointScalarType>(ns[1]);
        Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes3 =
            cachedChebnodes1d<PointScalarType >(ns[2]);

	if(!box.isNull()) {	    
	    transformNodesToInterval(nodes1, box.min()[0], box.max()[0]);
	    transformNodesToInterval(nodes2, box.min()[1], box.max()[1]);
	    transformNodesToInterval(nodes3, box.min()[2], box.max()[2]);
	}

	Eigen::Array<PointScalarType, Eigen::Dynamic, Eigen::Dynamic> xdiff = computeDiffVector(ns[0], x.row(0), nodes1);
        Eigen::Array<PointScalarType, Eigen::Dynamic, Eigen::Dynamic> ydiff = computeDiffVector(ns[1], x.row(1), nodes2);
        Eigen::Array<PointScalarType, Eigen::Dynamic, Eigen::Dynamic> zdiff = computeDiffVector(ns[2], x.row(2), nodes3);

        // std::cout << "DIMPUT = " << DIMOUT << std::endl;
        // std::cout << "shape xdiff = " << xdiff.rows() << " " << xdiff.cols() << std::endl;
        // std::cout << "shape ydiff = " << ydiff.rows() << " " << ydiff.cols() << std::endl;
        // std::cout << "shape zdiff = " << zdiff.rows() << " " << zdiff.cols() << std::endl;

#ifdef USE_NGSOLVE	
        tmult.Start();
#endif
        size_t idx=0;

        Eigen::Array<T, Eigen::Dynamic, DIMOUT> mytmp(x.cols(), DIMOUT);
        mytmp.fill(0);
	
        //std::cout << "type diff = " << typeid(x(0,0)).name() << ", type vals = " << typeid(vals(0,0)).name() << std::endl;
        
        for (size_t i = 0; i < ns[2]; i++) {
            for (size_t j = 0; j < ns[1]; j++) {	       
		// we vectorize the innermost loop. This part computes
		// result+=\sum_{k} vals[i,j,k]*xdiff[l,:]ydiff[j,:] zdiff[i,:]
#ifdef USE_NGSOLVE
              tmultx.Start();
              tmultx.AddFlops(xdiff.cols()*xdiff.rows());
#endif
              // auto tmp = (xdiff.matrix() * vals.middleRows(idx, ns[0]).matrix())
              // .array(); // nom.matrix()).array();
              mytmp =  (xdiff.matrix() * vals.middleRows(idx, ns[0]).matrix()).array();
#ifdef USE_NGSOLVE	      
              tmultx.Stop();
              tmultyz.Start();
#endif
		for (int l = 0; l < DIMOUT; l++) {
		  result.col(l) += (zdiff.col(i) * ydiff.col(j)) * mytmp.col(l);
		}
#ifdef USE_NGSOLVE
              tmultyz.Stop();

#endif
		idx+=ns[0];
            }
        }
#ifdef USE_NGSOLVE
        tmult.Stop();
#endif
        
	Eigen::Array<PointScalarType, -1,1> w1 = xdiff.rowwise().sum().transpose();
        Eigen::Array<PointScalarType, -1,1> w2 = ydiff.rowwise().sum().transpose();
        Eigen::Array<PointScalarType, -1,1> w3 = zdiff.rowwise().sum().transpose();

	

        const auto denom = w1 * w2 * w3;
        result.colwise() /= denom;

        return result;
    }


    //2d evaluation code
    template <typename T, typename PointScalarType, int DIMOUT>
    inline Eigen::Array<T, Eigen::Dynamic, DIMOUT>
    evaluate(
	     const Eigen::Ref<const Eigen::Array<PointScalarType , 2, Eigen::Dynamic> > &x,
             const Eigen::Ref<const Eigen::Array<T , Eigen::Dynamic, DIMOUT> > &vals,
             const Eigen::Vector<int, 2>& ns,
             BoundingBox<2> box = BoundingBox<2>()  )
    {
        const int DIM = 2;
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(x.cols(), DIMOUT);

        result.fill(0);

        assert(DIM == x.rows());
        Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes1 =
            cachedChebnodes1d<PointScalarType>(ns[0]);        
        Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes2 =
            cachedChebnodes1d<PointScalarType>(ns[1]);

	if(!box.isNull()) {	    
	    transformNodesToInterval(nodes1, box.min()[0], box.max()[0]);
	    transformNodesToInterval(nodes2, box.min()[1], box.max()[1]);
	}

	Eigen::Array<PointScalarType, Eigen::Dynamic, Eigen::Dynamic> xdiff = computeDiffVector(ns[0], x.row(0), nodes1);
        Eigen::Array<PointScalarType, Eigen::Dynamic, Eigen::Dynamic> ydiff = computeDiffVector(ns[1], x.row(1), nodes2);

        size_t idx=0;
	for (size_t j = 0; j < ns[1]; j++) {	       
	    // we vectorize the innermost loop. This part computes
	    // result+=\sum_{k} vals[i,j,k]*xdiff[l,:]ydiff[j,:] zdiff[i,:]
	    auto tmp = (xdiff.matrix() * vals.middleRows(idx, ns[0]).matrix())
		.array(); // nom.matrix()).array();
	    for (int l = 0; l < DIMOUT; l++) {
		result.col(l) += (ydiff.col(j)) * tmp.col(l);
	    }
	    idx+=ns[0];	
        }

	Eigen::Array<PointScalarType, x.ColsAtCompileTime,1> w1 = xdiff.rowwise().sum().transpose();
        Eigen::Array<PointScalarType, x.ColsAtCompileTime,1> w2 = ydiff.rowwise().sum().transpose();
	

        const auto denom = w1 * w2;
        result.colwise() /= denom;

        return result;
    }

    template <typename T, unsigned int DIM,  char package,int... Ns>
    inline int __eval(const Eigen::Ref<const Eigen::Array<double, DIM,Eigen::Dynamic> >  &points,
		      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic,1> > &interp_values,
		      const Eigen::Vector<int,DIM>& ns,
		      Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,1> > dest, size_t i, size_t n_points)
    {
	const int DIMOUT=1;
	const unsigned int packageSize = 1 << package;
	const size_t np = n_points / packageSize;
	n_points = n_points % packageSize;
			 
		      
	Eigen::Array<double,DIM,packageSize> tmp;
	
	ChebychevInterpolation::ClenshawEvaluator<T, packageSize,  DIM,DIM, DIMOUT,Ns...> clenshaw;
	for (int j = 0; j < np; j++) {
	    tmp=points.middleCols(i, packageSize);	    
	    dest.segment(i, packageSize) = clenshaw(tmp, interp_values,ns);
	    i += packageSize;
	}
	if constexpr(package > 0) {
	    if (n_points > 0) {
		i = __eval < T,  DIM,  package - 1,Ns... > (points, interp_values, ns, dest, i, n_points);
	    }
	}

	return i;

    }

    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    void fast_evaluate_tp(
				 const Eigen::Ref<const Eigen::Array<double, DIM-1, Eigen::Dynamic> >
				  &points,
                                  const Eigen::Ref<const Eigen::Array<double, 1, Eigen::Dynamic> >
				  &points2,
                                  int axis,
				  const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
				  &interp_values,
				  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
				  const Eigen::Vector<int, DIM>& ns,
				 BoundingBox<DIM> box = BoundingBox<DIM>());
            

    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    void tp_evaluate(
		     const std::array< Eigen::Array<double, Eigen::Dynamic,1> , DIM >
		     &points,					 
		     const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
		     &interp_values,
		     Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
		     const Eigen::Vector<int, DIM>& ns );

    
    

    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    void parallel_evaluate(
				  const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >
				  &points,
				  const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
				  &interp_values,
				  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
				  const Eigen::Vector<int, DIM>& ns,
				  BoundingBox<DIM> box = BoundingBox<DIM>());

    
    template<int DIM>
    inline size_t cache_key(const Eigen::Ref< const Eigen::Vector<int, DIM> >& ns)
    {
	size_t val=0;
	for(int i=0;i<DIM;i++){
	    val+=ns[i];
	    val=val<<8;
	}
	return val;
    }



    

    template <typename T, int DIM>
    const Eigen::Array<T, DIM, Eigen::Dynamic> &chebnodesNdd( const Eigen::Ref< const Eigen::Vector<int, DIM> >& ns)
    {
        static std::unordered_map<size_t, Eigen::Array<T, DIM, Eigen::Dynamic> >
            cache;

	static tbb::spin_mutex mutex;

	const size_t key=cache_key(ns);
	if (cache.count(key) > 0) {
	    return cache[key];
	} else {
	    tbb::spin_mutex::scoped_lock lock(mutex);
	    if (cache.count(key) > 0) {
		return cache[key];
	    }

	    if constexpr(DIM==3) {
		cache[key] = chebnodesNd<T, -1, -1, -1>(ns);
	    }else if constexpr(DIM==2){
		    cache[key] = chebnodesNd<T, -1, -1>(ns);
	    }else {
		assert(DIM==1);
		cache[key] = chebnodesNd<T, -1>(ns);
	    }
	    
	    return cache[key];
	}        
    }

    template<typename T,int DIM, int DIMOUT>
    struct InterpolationData {
	Eigen::Vector<int, DIM> order;
	ConeDomain<DIM> grid;
	Eigen::Array<T, Eigen::Dynamic, DIMOUT> values;

	size_t computeStride () const
	{
	    return order.prod();
	}
    };


};

#include "chebinterp.cpp"
#endif
