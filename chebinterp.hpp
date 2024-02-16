#ifndef _CHEBINTERP_HPP_
#define _CHEBINTERP_HPP_

#include <tuple>

#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <exception>


namespace ChebychevInterpolation
{

    template< typename T,size_t n>
    inline auto chebnodes1d() 
    {
	const double shift=M_PI/(2*n); //0= second kind pi/2n=first kind
	return ( Eigen::Array<T,n,1>::LinSpaced(n,shift,M_PI-shift)).cos();
    }


    template<typename T>
    struct InterpolationData {
	unsigned int order;
	int nh;
	Eigen::Array<T, Eigen::Dynamic, 1> values;	
    };


    template< typename T,size_t n1d, int DIM>
    inline Eigen::Array<T,DIM,Eigen::Dynamic>  chebnodesNd()
    {
	const size_t N=std::pow(n1d,DIM);
	auto nodes1d=chebnodes1d<T,n1d>();


	Eigen::Array<T,DIM,Eigen::Dynamic> nodesNd(DIM ,N);
	if constexpr(DIM==1){
	    nodesNd=nodes1d;
	}
	if constexpr(DIM==2){
	    for(size_t i=0;i<n1d;++i)
	    {
		for (size_t j=0;j<n1d;j++) {
		    nodesNd(0,i*n1d+j)=nodes1d[j];
		    nodesNd(1,i*n1d+j)=nodes1d[i];
		}
	    }
	}
	else{
	    Eigen::Array<T,3,Eigen::Dynamic> nodes3d;
	    for(size_t i=0;i<n1d;++i)
	    {
		for (size_t j=0;j<n1d;j++) {
		    for (size_t k=0;k<n1d;k++) {
			nodesNd(0,i*n1d*n1d+j*n1d+k)=nodes1d[k];
			nodesNd(1,i*n1d*n1d+j*n1d+k)=nodes1d[j];
			nodesNd(2,i*n1d*n1d+j*n1d+k)=nodes1d[i];
		    }
		}
	    }

	}
	return nodesNd;
    }

    bool iszero(double z)
    {
	return z==0;
    }


    bool isfinite(double z)
    {
	return std::isfinite(z);
    }
    
    bool iszero(std::complex<double> z)
    {
	return z.real()==0 && z.imag()==0;
    }

    bool isfinite(std::complex<double> z)
    {
	return std::isfinite(z.real()) && std::isfinite(z.imag());
    }


    template <typename T,int N_POINTS_AT_COMPILE_TIME, size_t n,unsigned int DIM,typename Derived1,typename Derived2>
    inline Eigen::Array<T,N_POINTS_AT_COMPILE_TIME,1> evaluate(const Eigen::ArrayBase<Derived1>&  x, const Eigen::ArrayBase<Derived2>& vals)
    {
	Eigen::Array<T,N_POINTS_AT_COMPILE_TIME,1> result(x.cols());
	Eigen::Array<T,N_POINTS_AT_COMPILE_TIME,1> nom(x.cols());
	Eigen::Array<T,N_POINTS_AT_COMPILE_TIME,1> weight(x.cols());
	result.fill(0);
	nom.fill(0);
	weight.fill(0);

	Eigen::Array<T,N_POINTS_AT_COMPILE_TIME,1> exact(x.cols());
		
	const auto& nodes=chebnodes1d<T,n>();

	assert(DIM==x.rows());
	const size_t stride=std::pow(n,DIM-1);

	int sign=1;
	for(size_t j=0;j<n;j++) {
	    auto xdiff=(x.row(DIM-1)-nodes[j]).transpose();
	    T cj=sign;
	    sign*=-1;
	    if(j==0 || j==n-1) {
		cj*=0.5;
	    }


	    
	    if constexpr (DIM==1) {
		const T fj=vals(j);
		nom+=cj*fj/xdiff;

		for(size_t l=0;l<x.cols();l++) {
		    if(iszero(xdiff[l])) {
			exact[l]=fj;
		    }
		}

		    
	    }else{
		auto fj=evaluate<T,N_POINTS_AT_COMPILE_TIME,n,DIM-1>(x.topRows(DIM-1),vals.segment(j*stride,stride));
	
		for(size_t l=0;l<x.cols();l++) {
		    if(iszero(xdiff[l])) {
			exact[l]=fj[l];
		    }
		}

		
		nom+=cj*fj/xdiff;
		
	    }
	    weight+=cj/xdiff;
	    
	}

	result=nom/weight;


	for(size_t l=0;l<x.cols();l++) {
	    if(!isfinite(result[l])) {
		result[l]=exact[l];
	    }
	}



	return result;
    }


    template <typename T, size_t n,unsigned int DIM,char package,typename T1,typename T2,typename T3>
    inline int __eval(const T1&  points,
		      const T2& interp_values,
		      T3& dest,size_t i,size_t n_points)
    {
	const unsigned int packageSize=1<<package;	    
	const size_t np=n_points / packageSize;
	n_points=n_points % packageSize;
	
	//std::cout<<"eval"<<packageSize<<" "<<np<<" "<<n_points<<std::endl;
	for(int j=0;j<np;j++){
	    dest.segment(i,packageSize) =ChebychevInterpolation::evaluate<T,packageSize,n,DIM>(points.middleCols(i,packageSize),interp_values);
	    i+=packageSize;
	}
	if constexpr(package>0) {
	    if(n_points>0)
		i=__eval<T,n,DIM,package-1>(points,interp_values,dest,i,n_points);
	}
	
	
	return i;
    	
    }



    template <typename T, size_t n,unsigned int DIM>
    inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double,DIM,Eigen::Dynamic,Eigen::RowMajor> >&  points,
				  const Eigen::Ref<const Eigen::Array<T,Eigen::Dynamic,1> >& interp_values,
				  Eigen::Ref<Eigen::Array<T,Eigen::Dynamic,1> > dest)
    {
	auto partial_loop =   [&](tbb::blocked_range<size_t>(r)){
	    size_t i=r.begin();
	    size_t n_points=r.end()-r.begin();
	    //We do packages of size 16, 8, 4, 2, 1	    
	    i=__eval<T,n,DIM,2>(points,interp_values,dest,i,n_points);
	    //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
	    assert(i==r.end());	    
	};

	tbb::parallel_for(tbb::blocked_range<size_t>(0,points.cols(),64),
			  partial_loop);
	//partial_loop(tbb::blocked_range<size_t>(0,points.cols(),64));
    }


    template<auto begin, auto end>
    inline void unroll(auto foo)
    {
      if constexpr(begin < end)
      {
	foo.template operator()<begin>();
	unroll<begin + 1, end>(foo);
      }
    }



    template <typename T, unsigned int DIM>
    inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double,DIM,Eigen::Dynamic,Eigen::RowMajor> >&  points,
				  const Eigen::Ref<const Eigen::Array<T,Eigen::Dynamic,1> >& interp_values,
				  Eigen::Ref<Eigen::Array<T,Eigen::Dynamic,1> > dest, int n)
    {
	const int NUM_SPECIALIZATIONS=40;
	static std::array<std::function<void (const Eigen::Ref<const Eigen::Array<double,DIM,Eigen::Dynamic,Eigen::RowMajor> >&,
					      const Eigen::Ref<const Eigen::Array<T,Eigen::Dynamic,1> >&,
					      Eigen::Ref<Eigen::Array<T,Eigen::Dynamic,1> >) >, NUM_SPECIALIZATIONS> lut;


	unroll<0,NUM_SPECIALIZATIONS>([&]<int NUM>()
				      {
					  lut[NUM] =parallel_evaluate<T,2*NUM,DIM>;
				      });


	assert(n/4<NUM_SPECIALIZATIONS);

	return lut[n/4](points, interp_values, dest);
    }


    template< typename T, int DIM>
    inline Eigen::Array<T,DIM,Eigen::Dynamic>  chebnodesNdd(unsigned int n)
    {
	const int NUM_SPECIALIZATIONS=30;
	static std::array<std::function<Eigen::Array<T,DIM,Eigen::Dynamic> () >, NUM_SPECIALIZATIONS> lut;


	unroll<0,NUM_SPECIALIZATIONS>([&]<int NUM>()
				      {
					  lut[NUM] =chebnodesNd<T,2*NUM,DIM>;
				      });

	assert(n/4<NUM_SPECIALIZATIONS);


	return lut[n/4]();


    }



};

#endif
