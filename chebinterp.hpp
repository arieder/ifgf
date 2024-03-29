#ifndef _CHEBINTERP_HPP_
#define _CHEBINTERP_HPP_

#include "boundingbox.hpp"
#include <tuple>

#include <fenv.h>
#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <exception>
#include<iostream>

#include "cone_domain.hpp"


namespace ChebychevInterpolation
{
    template< typename T, int N_AT_COMPILE_TIME>
    inline constexpr Eigen::Array<T,N_AT_COMPILE_TIME,1> chebnodes1d(int n)
    {
        assert(N_AT_COMPILE_TIME==-1 || n==N_AT_COMPILE_TIME);
        std::cout<<"new chebnodes"<<std::endl;
        /*//std::cout<<"a="<<n<<std::endl;
          Eigen::Array<T,n,1> pnts;
          //std::cout<<"b"<<std::endl;
          for(int i=0;i<n;i++) {
          //std::cout<<i<<std::endl;;
          pnts[i]=std::cos(M_PI*static_cast<double>(i)/(static_cast<double>(n)-1.0));
          }
          //std::cout<<"c"<<std::endl;

          return pnts;*/

        return (Eigen::Array<T, N_AT_COMPILE_TIME, 1>::LinSpaced((int) n, 0, M_PI)).cos(); 
    }

    template< typename T, int n>
    inline constexpr auto chebnodes1d_2()
    {
        Eigen::Array<T,n,1> pnts;
        for(int i=0;i<n;i++) {
            pnts[i]=cos(M_PI*(2.0*i+1.0)/(2.0*n));
        }
        return pnts;
        //const double shift = M_PI / (2 * n); //0= second kind pi/2n=first kind
        //return (Eigen::Array<T, n, 1>::LinSpaced(n, 0, M_PI)).cos();
    }
    //Represents a uniform refinement of (-1,1)^d


constexpr size_t order_for_dim(size_t order, size_t DIM)
{
    /*if (DIM==0){
        return order;
    }
    if (DIM==1) {
        return order;
    }
    if(DIM==2) {
        return std::round(2*order);
	}*/
    return order;
}

template< typename T,  int N1D_COMPILE_TIME, int DIM>
inline Eigen::Array<T, DIM, Eigen::Dynamic>  chebnodesNd(int n1d)
{
    const Eigen::Array<T,N1D_COMPILE_TIME,1> nodes1d = chebnodes1d<T, N1D_COMPILE_TIME >(n1d);


    if constexpr(DIM==1){
        Eigen::Array<T, DIM, Eigen::Dynamic>  nodesNd = nodes1d.transpose();
        return nodesNd;
    }else if constexpr (DIM==2) {
        auto nodesr = chebnodes1d<T, N1D_COMPILE_TIME>(n1d);
	Eigen::Array<T, DIM, Eigen::Dynamic> nodesNd(DIM, nodes1d.size()*nodesr.size());

        for (size_t i = 0; i < order_for_dim(n1d,2); ++i) {
            for (size_t j = 0; j < order_for_dim(n1d,1); j++) {
                nodesNd(0, i * order_for_dim(n1d,1) + j) = nodes1d[j];
                nodesNd(1, i * order_for_dim(n1d,1) + j) = nodesr[i];
            }
        }
        return nodesNd;
    }else if constexpr(DIM==3) {
        Eigen::Array<T, DIM, Eigen::Dynamic> nodesNd(DIM, (int) std::pow(n1d,3));
        for (size_t i = 0; i < n1d; ++i) {
            for (size_t j = 0; j < n1d; j++) {
                for (size_t k = 0; k < n1d; k++) {
                    nodesNd(0, i * n1d * n1d + j * n1d + k) = nodes1d[k];
                    nodesNd(1, i * n1d * n1d + j * n1d + k) = nodes1d[j];
                    nodesNd(2, i * n1d * n1d + j * n1d + k) = nodes1d[i];
                }
            }
        }
        return nodesNd;
    }else{
        std::cout << "not implemented yet" << std::endl;
        Eigen::Array<T, DIM, Eigen::Dynamic>  nodesNd;
        return nodesNd;
    }

}

bool iszero(double z)
{
    return z == 0;
}

bool isfinite(double z)
{
    return std::isfinite(z);
}

bool iszero(std::complex<double> z)
{
    return z.real() == 0 && z.imag() == 0;
}

bool isfinite(std::complex<double> z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

    template <typename T, int N_POINTS_AT_COMPILE_TIME, int N_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT, typename Derived1, typename Derived2>
inline Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> evaluate_slow(const Eigen::ArrayBase<Derived1>  &x,
                                                                  const Eigen::ArrayBase<Derived2> &vals,
                                                                  const Eigen::Ref<const
                                                                  Eigen::Vector<double, N_AT_COMPILE_TIME> >& nodes,
                                                                  int n )
{
    assert(N_AT_COMPILE_TIME==-1 || n==N_AT_COMPILE_TIME);
    //std::cout<<"slow"<<std::endl;
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> result(x.cols());
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> nom(x.cols());
    Eigen::Array<double, N_POINTS_AT_COMPILE_TIME, 1> weight(x.cols());
    result.fill(0);
    nom.fill(0);
    weight.fill(0);
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> exact(x.cols());
    

    assert(DIM == x.rows());
    const size_t stride = std::pow(n, DIM - 1);
    int sign = 1;
    for (size_t j = 0; j < nodes.size(); j++) {
        auto xdiff = (x.row(DIM - 1) - nodes[j]).transpose();
        T cj = sign;//*sin((2.0*j+1.0)*M_PI/(2.0*n));
        sign *= -1;
        if (j == 0 || j == nodes.size()-1) {
            cj *= 0.5;
	}

        if constexpr(DIM == 1) {
            const T fj = vals(j);
            nom += cj * fj / xdiff;

            for (size_t l = 0; l < x.cols(); l++) {
                if (iszero(xdiff[l])) {
                    exact[l] = fj;
                }
            }

        } else {
            auto fj = evaluate_slow < T, N_POINTS_AT_COMPILE_TIME, N_AT_COMPILE_TIME, DIM - 1,DIMOUT > (x.topRows(DIM - 1), vals.segment(j * stride, stride),nodes,n);

            for (size_t l = 0; l < x.cols(); l++) {
                if (iszero(xdiff[l])) {
                    exact[l] = fj[l];
                }
            }

            nom += cj * fj / xdiff;

        }
        weight += cj / xdiff;

    }

    result = nom / weight;

    for (size_t l = 0; l < x.cols(); l++) {
        if (!isfinite(result[l])) {
            result[l] = exact[l];
        }
    }

    return result;
}


    
    template <typename T, int N_POINTS_AT_COMPILE_TIME, int N_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT, typename Derived1, typename Derived2>
inline Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, DIMOUT> evaluate(const Eigen::ArrayBase<Derived1>  &x,
                                                             const Eigen::ArrayBase<Derived2> &vals,
                                                             const Eigen::Ref<const Eigen::Vector<double, N_AT_COMPILE_TIME > >& nodes,
                                                             int n)
{
    assert(N_AT_COMPILE_TIME==-1 || n==N_AT_COMPILE_TIME);
    if constexpr(DIM!=3){
        static_assert(DIMOUT==1, "only scalar output supported at this point");
	return evaluate_slow<T,N_POINTS_AT_COMPILE_TIME,N_AT_COMPILE_TIME,DIM,DIMOUT>(x,vals,nodes,n);
    }

    assert(vals.cols()==DIMOUT);
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME,DIMOUT> result(x.cols(),DIMOUT);

    Eigen::Array<double, N_POINTS_AT_COMPILE_TIME,DIM> weight(x.cols(),DIM);
    result.fill(0);
    weight.fill(0);

    assert(DIM == x.rows());
    if(nodes.size()!=n) {
	std::cout<<"wtf?"<<nodes.size()<<" "<<n<<std::endl;
    }
    assert(nodes.size()==n);
    assert(vals.size()==pow(n,DIM));

    Eigen::Array<double, N_AT_COMPILE_TIME,1 > c(n,1);
    for(size_t i=0;i<n;i++){
	c[i]=(i % 2 == 0) ? 1.0:-1.0;
    }
    c[0]*=0.5;
    c[n-1]*=0.5;

    Eigen::Array<double, N_POINTS_AT_COMPILE_TIME,N_AT_COMPILE_TIME> xdiff(x.cols(),n);
    //xdiff=(1.0/((-nodes.array()).replicate(1,n).rowwise() + x.row(0)+1e-12).transpose()).rowwise()*c.transpose();
    for(int i=0;i<x.cols();i++) {
	xdiff.row(i)=c*1.0/(x(0,i)-nodes.array()+1e-12);
    }


    Eigen::Array<double, N_POINTS_AT_COMPILE_TIME,N_AT_COMPILE_TIME> ydiff(x.cols(),n);//=x.row(1).colwise()-nodes;
    //ydiff=(1.0/((-nodes.array()).replicate(1,n).rowwise() + x.row(1)+1e-12).transpose()).rowwise()*c.transpose();
    for(int i=0;i<x.cols();i++) {
	ydiff.row(i)=c*1.0/(x(1,i)-nodes.array()+1e-12);
    }

    Eigen::Array<double, N_POINTS_AT_COMPILE_TIME,N_AT_COMPILE_TIME> zdiff(x.cols(),n);//=x.row(1).colwise()-nodes;
    //zdiff=(1.0/((-nodes.array()).replicate(1,n).rowwise() + x.row(2)+1e-12).transpose()).rowwise()*c.transpose();
    //zdiff=1.0/(x.row(2).replicate(n,1)-nodes.array()).transpose();
    
    for(int i=0;i<x.cols();i++) {
	zdiff.row(i)=c*1.0/(x(2,i)-nodes.array()+1e-12);
    }

    //Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> tmp(x.cols(),1);
    //Eigen::Array<T, n, 1> nom(n,1);        

    const size_t strideI = n*n;
    const size_t strideJ = n;
    int sign = 1;
    for(size_t i=0;i<n;i++) {
	for(size_t j=0;j<n;j++) {
	    const size_t idx=i*strideI+j*strideJ;
	    //we vectorize the innermost loop. This part computes
            //result+=\sum_{k} vals[i,j,k]*xdiff[l,:]ydiff[j,:] zdiff[i,:]
	    auto tmp=(xdiff.matrix()*vals.middleRows(idx,n).matrix()).array(); //nom.matrix()).array();
	    for(int l=0;l<DIMOUT;l++) {
		result.col(l)+=(zdiff.col(i) * ydiff.col(j))*tmp.col(l);
	    }

	    //std::cout<<"done"<<std::endl;
		/*for(size_t l=0;l<n;l++) {



		double w=wi*wj*wl;

		assert(i<xdiff.cols());
		assert(j<ydiff.cols());
		assert(l<zdiff.cols());

		const T f = vals(idx);

		result += w * f / (xdiff.col(l)*ydiff.col(j)*zdiff.col(i));*/

	/*if(result.hasNaN())
	{
	    std::cout<<"res="<<result<<std::endl;
	    std::cout<<xdiff.col(i)<<std::endl;
	    std::cout<<ydiff.col(i)<<std::endl;
	    std::cout<<zdiff.col(i)<<std::endl;
	    }*/
	//assert(!result.hasNaN());
	//}
	}
    }

    for(size_t i=0;i<nodes.size();i++)
    {
	weight.col(0)+=xdiff.col(i).transpose();
	weight.col(1)+=ydiff.col(i).transpose();
        weight.col(2)+=zdiff.col(i).transpose();

    }
    auto denom=weight.col(0)*weight.col(1)*weight.col(2);
    result.colwise()/=denom;

    return result;
}






template <typename T, long int n, unsigned int DIM, char package, typename T1, typename T2, typename T3>
inline int __eval(const T1  &points,
                  const T2 &interp_values,
                  T3 &dest, size_t i, size_t n_points,const Eigen::Ref<const Eigen::Vector<double, n> >& nodes)
{
    const unsigned int packageSize = 1 << package;
    const size_t np = n_points / packageSize;
    n_points = n_points % packageSize;

    //std::cout<<"eval"<<packageSize<<" "<<np<<" "<<n_points<<std::endl;
    for (int j = 0; j < np; j++) {
        dest.segment(i, packageSize) = ChebychevInterpolation::evaluate_slow<T, packageSize, n, DIM>(points.middleCols(i, packageSize), interp_values,nodes);
        i += packageSize;
    }
    if constexpr(package > 0) {
        if (n_points > 0) {
            i = __eval < T, n, DIM, package - 1 > (points, interp_values, dest, i, n_points,nodes);
        }
    }

    return i;

}


template <typename T, int N_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT>
inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >  &points, 
                              const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> > &interp_values,
                              Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,size_t n)
{
    static Eigen::Vector<double,N_AT_COMPILE_TIME> nodes = chebnodes1d<double, N_AT_COMPILE_TIME>(n);

    if constexpr(N_AT_COMPILE_TIME==-1) {
	if(nodes.size()!=n) {
	    nodes=chebnodes1d<double, N_AT_COMPILE_TIME>(n);
	}
    }
    
    
    //fedisableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    switch(points.cols()) {
    case 1: dest=evaluate < T, 1, N_AT_COMPILE_TIME, DIM, DIMOUT > (points, interp_values,nodes,n); break;
	//case 2: dest=evaluate < T, 2, n, DIM > (points, interp_values,nodes); break;
	//    case 3: dest=evaluate < T, 3, n, DIM > (points, interp_values,nodes); break;
	//case 4: dest=evaluate < T, 4, n, DIM > (points, interp_values,nodes); break;
	//case 5: dest=evaluate < T, 5, n, DIM > (points, interp_values,nodes); break;
			
    default:
	dest=evaluate<T, Eigen::Dynamic, N_AT_COMPILE_TIME, DIM,DIMOUT>(points, interp_values,nodes,n);
    }

    //dest=ChebychevInterpolation::evaluate<T, Eigen::Dynamic, n, DIM>(points, interp_values,nodes);
    
    //return evaluate_slow < T, Eigen::Dynamic, n, 0 > (x.topRows(0), tmp.segment(i * stride, stride),nodes);
    
    
    /*auto partial_loop =   [&](tbb::blocked_range<size_t>(r)) {
        size_t i = r.begin();
        size_t n_points = r.end() - r.begin();
        //We do packages of size 4, 2, 1
        i = __eval<T, n, DIM, 0>(points, interp_values, dest, i, n_points,nodes);
        //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
        assert(i == r.end());
	};
    */	
    //tbb::parallel_for(tbb::blocked_range<size_t>(0, points.cols(), 64),
    //                 partial_loop);
    //partial_loop(tbb::blocked_range<size_t>(0,points.cols()));
    
    //feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    
}

template<auto begin, auto end>
inline void unroll(auto foo)
{
    if constexpr(begin < end) {
        foo.template operator()<begin>();
        unroll < begin + 1, end > (foo);
    }
}

#define NUM_SPECIALIZATIONS  (unsigned int) 10
template <typename T, unsigned int DIM, unsigned int DIMOUT>
inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic > >  &points,
                              const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> > &interp_values,
                              Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest, int n)
{
    if(n<=NUM_SPECIALIZATIONS) {
        static std::array<std::function<void
                                        (const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >&,
                                         const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >&,
                                         Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> >,
                                         size_t) >,NUM_SPECIALIZATIONS> lut;
    
        unroll<0, NUM_SPECIALIZATIONS>([&]<int NUM>() {
                lut[NUM] = parallel_evaluate<T, NUM+1, DIM,DIMOUT>;
            });

        lut[ n-1 ](points, interp_values, dest,n);
    }else {
        return parallel_evaluate<T, Eigen::Dynamic, DIM, DIMOUT>(points,interp_values,dest,n);
    }

    //return parallel_evaluate<T, 10, DIM>(points,interp_values,dest);
}

template< typename T, int DIM>
inline const Eigen::Array<T, DIM, Eigen::Dynamic>&  chebnodesNdd(unsigned int n)
{    
    static std::array<Eigen::Array<T, DIM, Eigen::Dynamic>, NUM_SPECIALIZATIONS> lut {
	chebnodesNd<T,  1, DIM>(1),
	chebnodesNd<T,  2, DIM>(2),
	chebnodesNd<T,  3, DIM>(3),
	chebnodesNd<T,  4, DIM>(4),
	chebnodesNd<T,  5, DIM>(5),
	chebnodesNd<T,  6, DIM>(6),
	chebnodesNd<T,  7, DIM>(7),
	chebnodesNd<T,  8, DIM>(8),
	chebnodesNd<T,  9, DIM>(9),
	chebnodesNd<T,  10, DIM>(10),
    };
	
    if(n<=NUM_SPECIALIZATIONS) {
        return lut[ n-1 ];
    }

    static std::unordered_map<size_t, Eigen::Array<T, DIM, Eigen::Dynamic > > cache;

    if(cache.contains(n)) {
        return cache[n];
    }else{       
        cache[n]=chebnodesNd<T,-1, DIM>(n);
        return cache[n];
    }
}

    
    template<typename T,int DIM, int DIMOUT>
struct InterpolationData {
    unsigned int order;
    ConeDomain<DIM> grid;
    Eigen::Array<T, Eigen::Dynamic, DIMOUT> values;
};


};

#endif
