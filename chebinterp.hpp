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

#include <sycl/sycl.hpp>
using namespace sycl;

namespace ChebychevInterpolation
{

    template< typename T, int N_AT_COMPILE_TIME>
    inline constexpr Eigen::Array<T,N_AT_COMPILE_TIME,1> chebnodes1d(int n)
    {
	std::cout<<"new chebnodes"<<n<<std::endl;
        assert(N_AT_COMPILE_TIME==-1 || n==N_AT_COMPILE_TIME);
        return (Eigen::Array<T, N_AT_COMPILE_TIME, 1>::LinSpaced((int) n, 0, M_PI)).cos(); 
    }


    template< typename T>
    Eigen::Array<T,Eigen::Dynamic,1> cachedChebnodes1d(int n)
    {
	static std::unordered_map<size_t, Eigen::Array<T, Eigen::Dynamic,1 > >  cache;
	if (cache.contains(n)) {
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

    int
    inline bary_weight(size_t i, size_t p) 
    {
	return (i==0 || i==p-1 ? 2:1) * (i % 2 == 0 ? 1:-1);
    }



    template <typename T, typename PointScalarType>
    void
    evaluate_sycl(
		  buffer<PointScalarType>& b_x,
		  buffer<T>& b_vals,
		  const mint3& ns,
		  buffer<PointScalarType>& b_nodes1,
		  buffer<PointScalarType>& b_nodes2,
		  buffer<PointScalarType>& b_nodes3,		  
		  buffer<T> & b_result,
		  handler& h,
		  size_t N)

    {	
    	accessor x(b_x,h,read_only);
	accessor vals(b_vals,h,read_only);

	accessor result(b_result,h,write_only);

	accessor nodes1(b_nodes1,h,read_only);
	accessor nodes2(b_nodes2,h,read_only);
	accessor nodes3(b_nodes3,h,read_only);


	h.parallel_for(N, [=](id<1> pnt)
	{
	    PointScalarType weight=0;
	    T f=0;
	    size_t idx=0;
	    for (size_t i = 0; i < ns[2]; i++) {
		const double v1=bary_weight(i,ns[2])* (x[pnt*3+2]-nodes1[i]-1e-16);
		for (size_t j = 0; j < ns[1]; j++) {
		    const double v2=v1*bary_weight(j,ns[1])*(x[pnt*3+1]-nodes2[j]-1e-16);		    
		    for(size_t l=0; l<ns[0]; l++) {
			const double v3=v2*bary_weight(l,ns[0])*(x[pnt*3]-nodes3[l]-1e-16);
			const double wijk=1.0/(v3);
			f+=vals[idx]*wijk;
			weight+=wijk;
			idx++;
		    }
		}		
	    }
	    result[pnt]=f/weight;
	});
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

	Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes1 =
            cachedChebnodes1d<PointScalarType>(ns[0]);        

	Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes2 =
            cachedChebnodes1d<PointScalarType>(ns[1]);        

	Eigen::Array<PointScalarType, Eigen::Dynamic, 1> nodes3 =
            cachedChebnodes1d<PointScalarType>(ns[2]);        


	if(!box.isNull()) {	    
	    transformNodesToInterval(nodes1, box.min()[0], box.max()[0]);
	    transformNodesToInterval(nodes2, box.min()[1], box.max()[1]);
	    transformNodesToInterval(nodes3, box.min()[2], box.max()[2]);
	}

	buffer b_nodes1{nodes1};
	buffer b_nodes2{nodes2};
	buffer b_nodes3{nodes3};

	Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(x.cols(), DIMOUT);
	buffer<T> b_result(x.cols());


	buffer<PointScalarType> b_x{x.data(),range{(unsigned long) 3*x.cols()}};
	buffer b_vals{vals};
	
	
	queue q;

	mint3 mns{ns[0],ns[1],ns[2]};
	q.submit([&](handler& h)
	{	   
	    evaluate_sycl(b_x,b_vals,mns,b_nodes1,b_nodes2,b_nodes3,b_result,h,x.cols());
	}
	    );


	//TODO more efficient way to avoid this extra copy?
	host_accessor result_ac(b_result,read_only);

	for(size_t i=0;i<x.cols();i++) {
	    result[i]=result_ac[i];
	}

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
        dest = evaluate(points, interp_values, ns,box);
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
	if (cache.contains(key)) {
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
