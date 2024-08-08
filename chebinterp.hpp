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
#include<iostream>

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

    bool iszero(double z) { return z == 0; }

    bool isfinite(double z) { return std::isfinite(z); }

    bool iszero(std::complex<double> z) {
        return z.real() == 0 && z.imag() == 0;
    }

    bool isfinite(std::complex<double> z) {
        return std::isfinite(z.real()) && std::isfinite(z.imag());
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
    inline void chebtransform(const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1> >& src,
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

	//std::cout<<"ns="<<ns<<std::endl;
	//std::cout<<"nsigma="<<nsigma<<std::endl;
	if constexpr(DIM==1) {	   
	    //do a straight forwad summation of the innermost dimension
	    for(size_t idx=0;idx<ns[0];idx++) {
		for(size_t sigma=0;sigma<ns[0];sigma++)  {
		    const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[0]));
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
		    const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));
		    dest.segment(idx*stride,nsigma)+=M.segment(sigma*stride,nsigma)*Td;
		}	    
		dest.segment(idx*stride,nsigma)*=(idx ==0 ? 1:2 )*1./ns[DIM-1];
	    }

	    //std::cout<<"err sum_factor="<<(D-dest).matrix().norm()<<std::endl;
	}
    }


    
    template <typename T, int POINTS_AT_COMPILE_TIME, int DIM, unsigned int DIMOUT,  typename Derived1, typename Derived2>
    inline Eigen::Array<T,POINTS_AT_COMPILE_TIME,DIMOUT> evaluate_clenshaw(const Eigen::ArrayBase<Derived1>  &x,
		const Eigen::ArrayBase<Derived2> &vals,
		const Eigen::Ref<const Eigen::Vector<int,DIM> >& ns )
    {
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> b1(x.cols(),DIMOUT);
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> b2(x.cols(),DIMOUT);
	Eigen::Array<T, POINTS_AT_COMPILE_TIME, DIMOUT> tmp(x.cols(),DIMOUT);


	if constexpr (DIM==1)
	{	    
	    b1.fill(0.);
	    b2.fill(0.);

	    for(size_t j=ns[0]-1;j>0;j--) {
		tmp=(2.*((b1)*x.row(0).transpose())-(b2))+vals(j);

		b2=b1;
		b1=tmp;
	    }
	    
	    return (b1*x.row(0).transpose()-b2)+vals(0);

	}else //recurse down
	{
	    
	    b1.fill(0);
	    b2.fill(0);
	    const size_t stride = ns.head(DIM-1).prod();

	    
	    auto c0=evaluate_clenshaw<T, POINTS_AT_COMPILE_TIME, DIM-1,DIMOUT>(x.topRows(DIM - 1),
									  vals.segment(0 * stride, stride),
									       ns.template head<DIM-1>()).eval();

	    for(size_t j=ns[0]-1;j>0;j--) {
		tmp= evaluate_clenshaw<T, POINTS_AT_COMPILE_TIME, DIM-1,DIMOUT>(x.topRows(DIM - 1),
										   vals.segment(j * stride, stride),
										   ns.template head<DIM-1>());
		tmp+=(2.*(b1*x.row(DIM-1).transpose())-b2);
		b2=b1;
		b1=tmp;
	    }

	    return (b1*x.row(DIM-1).transpose()-b2) + c0;

	}
    }	
    


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

    template <typename T, unsigned int DIM, char package, typename T1, typename T2, typename T3>
    inline int __eval(const T1  &points,
		      const T2 &interp_values,
		      const Eigen::Vector<int,DIM>& ns,
		      T3 &dest, size_t i, size_t n_points)
    {
	const int DIMOUT=1;
	const unsigned int packageSize = 1 << package;
	const size_t np = n_points / packageSize;
	n_points = n_points % packageSize;
			 
		      
	Eigen::Array<T,DIM,packageSize> tmp;

	for (int j = 0; j < np; j++) {
	    tmp=points.middleCols(i, packageSize);
	    dest.segment(i, packageSize) = ChebychevInterpolation::evaluate_clenshaw<T, packageSize,  DIM, DIMOUT>(tmp, interp_values,ns);
	    i += packageSize;
	}
	if constexpr(package > 0) {
	    if (n_points > 0) {
		i = __eval < T,  DIM, package - 1 > (points, interp_values, ns, dest, i, n_points);
	    }
	}

	return i;

    }
    

    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    inline void parallel_evaluate(
				  const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >
				  &points,
				  const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
				  &interp_values,
				  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
				  const Eigen::Vector<int, DIM>& ns,
				  BoundingBox<DIM> box = BoundingBox<DIM>())
    {
	
	Eigen::Array<T,DIM,Eigen::Dynamic> points0(DIM,points.cols());

	const auto a=0.5*(box.max()-box.min()).array();
	const auto b=0.5*(box.max()+box.min()).array();

	if(!box.isNull()) {
	    points0.array()=(points.array().colwise()-b).colwise()/a;
	}else {
	    points0=points;
	}

	//std::cout<<"ev"<<nodes[DIM-1]<<std::endl;

	//dest.resize(DIMOUT,points.cols());
        
	//template <typename T, int N_POINTS_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT, typename Derived1, typename Derived2, int N_AT_COMPILE_TIME, int... OTHER_NS>

#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem ifgf cheb::eval");

	ngcore::RegionTimer reg(t);
	t.AddFlops (ns[0]*ns[1]*ns[2]*points.cols()*DIMOUT);
#endif

	for(int i=0;i<points.cols();)
	{
	    size_t n_points = points.cols();
	    //We do packages of size 4, 2, 1
	    i = __eval<T, DIM, 3>(points0, interp_values, ns, dest, i, n_points);
	    //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
	    //assert(i == r.end());
	}
	    
	//tbb::parallel_for(tbb::blocked_range<size_t>(0, points.cols(), 64),
	//                 partial_loop);
	//partial_loop(tbb::blocked_range<size_t>(0,points.cols()));
    
	
	//dest = ChebychevInterpolation::evaluate_clenshaw<T,-1,DIM,DIMOUT>(points0,interp_values,ns);
	/*for (int j = 0; j < points0.cols(); j++) {
	    //dest = ChebychevInterpolation::evaluate_clenshaw<T,-1,DIM,DIMOUT>(points,interp_values,ns);
	    dest(j) = ChebychevInterpolation::evaluate_clenshaw<T,1,DIM,DIMOUT>(points0.col(j),interp_values,ns)[0];
	    //ChebychevInterpolation::evaluate_slow<T, 1,  DIM,1>(points.col(j), interp_values,nodes,ns)[0];	    
	    }*/

	//std::cout<<"done ev"<<std::endl;

    }

    
    template<int DIM>
    inline size_t cache_key(const Eigen::Vector<int, DIM>& ns)
    {
	size_t val=0;
	for(int i=0;i<DIM;i++){
	    val+=ns[i];
	    val=val<<8;
	}
	return val;
    }

    template <typename T, int DIM>
    inline const Eigen::Array<T, DIM, Eigen::Dynamic> &chebnodesNdd( const Eigen::Vector<int, DIM>& ns)
    {
        static std::unordered_map<size_t, Eigen::Array<T, DIM, Eigen::Dynamic> >
            cache;

	const size_t key=cache_key(ns);
	if (cache.count(key) > 0) {
	    return cache[key];
	} else {
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

#endif
