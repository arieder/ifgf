#ifndef _CHEBINTERP_HPP_
#define _CHEBINTERP_HPP_

#include <tuple>

#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <exception>

namespace ChebychevInterpolation
{
    template< typename T, size_t n>
inline constexpr auto chebnodes1d()
{
    /*//std::cout<<"a="<<n<<std::endl;
    Eigen::Array<T,n,1> pnts;
    //std::cout<<"b"<<std::endl;
    for(int i=0;i<n;i++) {
	//std::cout<<i<<std::endl;;
	pnts[i]=std::cos(M_PI*static_cast<double>(i)/(static_cast<double>(n)-1.0));
    }
    //std::cout<<"c"<<std::endl;

    return pnts;*/

    return (Eigen::Array<T, n, 1>::LinSpaced((int) n, 0, M_PI)).cos(); 
    }

    template< typename T, size_t n>
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
template<size_t DIM>
class ConeDomain
{
public:
    ConeDomain()
    {

    }

    ConeDomain(Eigen::Vector<size_t, DIM> numEls, const BoundingBox<DIM>& domain) :
	m_numEls(numEls),
	m_domain(domain)
    {
    }

    inline void setNElements(Eigen::Vector<size_t, DIM> numEls) {
	m_numEls=numEls;	
    }

    inline constexpr size_t n_elements() const
    {
	size_t n=1;
	for(size_t i=0;i<DIM;i++)
	    n*=m_numEls[i];
	return n;
    }
    

    inline const std::vector<size_t>& activeCones() const
    {
	return m_activeCones;
    }

    inline void setActiveCones( std::vector<size_t>& cones)
    {
	m_activeCones=cones;
    }


    inline void setConeMap( std::vector<size_t>& cone_map)
    {
	m_coneMap=cone_map;
    }


    inline size_t memId(size_t el) const
    {	
	return m_coneMap[el];
    }

    
    
    BoundingBox<DIM> domain() const {
	return m_domain;
    }

    //transforms from (-1,1) to K_el
    auto transform(size_t el,const Eigen::Ref<const Eigen::Matrix<double,DIM,Eigen::Dynamic> >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	for(size_t j=0;j<pnts.cols();j++)
	{
	    const auto a=0.5*(bbox.max()-bbox.min()).array();
	    const auto b=0.5*(bbox.max()+bbox.min()).array();

	    tmp.col(j)=pnts.col(j).array()*a+b;
	    //assert(bbox.contains(tmp.col(j)));
	}

	
	return tmp;
	    
    }



    //transforms from K_el to (-1,1)
    auto transformBackwards(size_t el,const Eigen::Ref<const Eigen::Matrix<double,DIM,Eigen::Dynamic> >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	for(size_t j=0;j<pnts.cols();j++)
	{	    
	    //assert(bbox.squaredExteriorDistance(pnts.col(j))/pnts.col(j).norm()<1e-12);
	    const auto a=0.5*(bbox.max()-bbox.min()).array();
	    const auto b=0.5*(bbox.max()+bbox.min()).array();
	    
	    tmp.col(j)=(pnts.col(j).array()-b)/a;

	    
	    //assert(-1-1e-8<=tmp.col(j)[0] && tmp.col(j)[0]<=1+1e-8);
	    //assert(-1-1e-8<=tmp.col(j)[1] && tmp.col(j)[1]<=1+1e-8);
	}
	
	return tmp;	    
    }


    BoundingBox<DIM> region(size_t j) const
    {
	auto j0=j;
	assert(j<n_elements());
	Eigen::Vector<double, DIM> min,max;
        Eigen::Vector<double, DIM> h=m_domain.diagonal();
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];

	    
	    
	    min[i]=m_domain.min()[i]+(idx*(h[i]/((double) m_numEls[i])));
	    max[i]=(min[i]+(h[i]/((double) m_numEls[i])));
	}

	return BoundingBox<DIM>(min,max);
    }

    size_t elementForPoint(Eigen::Vector<double,DIM> pnt) const
    {
	size_t idx=0;
	int stride=1;
	//std::cout<<"dom:"<<m_domain.min().transpose()<<" "<<m_domain.max().transpose()<<std::endl;
	//std::cout<<m_numEls<<std::endl;
	//std::cout<<"pn"<<pnt.transpose()<<std::endl;
	//assert(m_domain.squaredExteriorDistance(pnt)<1e-4);
	for(int j=0;j<DIM;j++) {	    
	    const double q=(pnt[j]-m_domain.min()[j])*m_numEls[j]/m_domain.diagonal()[j];
	    
	    //std::cout<<"floor"<<std::floor(q)<<std::endl;
	    const size_t ij=static_cast<int>( std::floor(std::clamp(q,0.0, m_numEls[j]-1.0)));
	    //std::cout<<q<<"ij "<<ij<<std::endl;
	    idx+=ij*stride;
	    stride*=m_numEls[j];
	}
	assert(idx<n_elements());
	return idx;

    }

    bool isEmpty() const
    {
	return m_domain.isEmpty();
    }
private:
    BoundingBox<DIM> m_domain;
    Eigen::Vector<size_t, DIM> m_numEls;
    std::vector<size_t> m_activeCones;
    std::vector<size_t> m_coneMap;
};

    

template<typename T,int DIM>
struct InterpolationData {
    unsigned int order;
    ConeDomain<DIM> grid;
    Eigen::Array<T, Eigen::Dynamic, 1> values;
};



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

template< typename T, size_t n1d, int DIM>
inline Eigen::Array<T, DIM, Eigen::Dynamic>  chebnodesNd()
{

    auto nodes1d = chebnodes1d<T, order_for_dim(n1d,1) >();


    if constexpr(DIM==1){
        Eigen::Array<T, DIM, Eigen::Dynamic>  nodesNd = nodes1d.transpose();
        return nodesNd;
    }else if constexpr (DIM==2) {
        auto nodesr = chebnodes1d<T, order_for_dim(n1d,2)>();
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

template <typename T, int N_POINTS_AT_COMPILE_TIME, size_t n, unsigned int DIM, typename Derived1, typename Derived2>
inline Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> evaluate(const Eigen::ArrayBase<Derived1>  &x, const Eigen::ArrayBase<Derived2> &vals)
{
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> result(x.cols());
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> nom(x.cols());
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> weight(x.cols());
    result.fill(0);
    nom.fill(0);
    weight.fill(0);
    Eigen::Array<T, N_POINTS_AT_COMPILE_TIME, 1> exact(x.cols());


    const auto &nodes = chebnodes1d<T, order_for_dim(n,DIM)>();

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
            auto fj = evaluate < T, N_POINTS_AT_COMPILE_TIME, n, DIM - 1 > (x.topRows(DIM - 1), vals.segment(j * stride, stride));

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

template <typename T, size_t n, unsigned int DIM, char package, typename T1, typename T2, typename T3>
inline int __eval(const T1  &points,
                  const T2 &interp_values,
                  T3 &dest, size_t i, size_t n_points)
{
    const unsigned int packageSize = 1 << package;
    const size_t np = n_points / packageSize;
    n_points = n_points % packageSize;

    //std::cout<<"eval"<<packageSize<<" "<<np<<" "<<n_points<<std::endl;
    for (int j = 0; j < np; j++) {
        dest.segment(i, packageSize) = ChebychevInterpolation::evaluate<T, packageSize, n, DIM>(points.middleCols(i, packageSize), interp_values);
        i += packageSize;
    }
    if constexpr(package > 0) {
        if (n_points > 0) {
            i = __eval < T, n, DIM, package - 1 > (points, interp_values, dest, i, n_points);
        }
    }

    return i;

}

template <typename T, size_t n, unsigned int DIM>
inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic, Eigen::RowMajor> >  &points,
                              const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1> > &interp_values,
                              Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > dest)
{
    auto partial_loop =   [&](tbb::blocked_range<size_t>(r)) {
        size_t i = r.begin();
        size_t n_points = r.end() - r.begin();
        //We do packages of size 16, 8, 4, 2, 1
        i = __eval<T, n, DIM, 4>(points, interp_values, dest, i, n_points);
        //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
        assert(i == r.end());
    };

    //tbb::parallel_for(tbb::blocked_range<size_t>(0, points.cols(), 64),
    //                 partial_loop);
    partial_loop(tbb::blocked_range<size_t>(0,points.cols()));
}

template<auto begin, auto end>
inline void unroll(auto foo)
{
    if constexpr(begin < end) {
        foo.template operator()<begin>();
        unroll < begin + 1, end > (foo);
    }
}

#define NUM_SPECIALIZATIONS  (unsigned int) 30
#define SHRINKING_FACTOR (unsigned int) 1
template <typename T, unsigned int DIM>
inline void parallel_evaluate(const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic, Eigen::RowMajor> >  &points,
                              const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1> > &interp_values,
                              Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > dest, int n)
{

    static std::array<std::function<void (const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic, Eigen::RowMajor> >&,
                                          const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1> >&,
                                          Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> >) >, NUM_SPECIALIZATIONS> lut;

    unroll<0, NUM_SPECIALIZATIONS>([&]<int NUM>() {
        lut[NUM] = parallel_evaluate<T,  SHRINKING_FACTOR*NUM, DIM>;
    });


    return lut[ n ](points, interp_values, dest);

    //return parallel_evaluate<T, 10, DIM>(points,interp_values,dest);
}

template< typename T, int DIM>
inline Eigen::Array<T, DIM, Eigen::Dynamic>  chebnodesNdd(unsigned int n)
{
    static std::array<std::function<Eigen::Array<T, DIM, Eigen::Dynamic> () >, NUM_SPECIALIZATIONS> lut;

    unroll<1, NUM_SPECIALIZATIONS>([&]<int NUM>() {
        lut[NUM] = chebnodesNd<T,  SHRINKING_FACTOR*NUM, DIM>;
    });


    return lut[n ]();
    
    //return chebnodesNd<T,10,DIM>();

}

};

#endif
