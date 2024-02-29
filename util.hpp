#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <numeric>

#include <algorithm>

namespace Util
{
template <class ExecutionPolicy, typename RandomIt, class Compare>
auto sort_with_permutation(ExecutionPolicy&& policy, RandomIt cbegin, RandomIt cend, Compare comp)
{
    auto len = std::distance(cbegin, cend);
    std::vector<size_t> perm(len);
    std::iota(perm.begin(), perm.end(), 0U);
    std::sort(policy, perm.begin(), perm.end(),
    [&](const size_t &a, const size_t &b) {
        return comp(*(cbegin + a), *(cbegin + b));
    });
    return perm;
}

template <typename T, int DIM>
Eigen::Matrix<T, DIM, Eigen::Dynamic> copy_with_permutation(const Eigen::Matrix<T, DIM, Eigen::Dynamic> &v, const std::vector<size_t> &permutation)
{
    Eigen::Matrix<T, DIM, Eigen::Dynamic> data(DIM, v.cols());
    for (size_t i = 0; i < v.cols(); i++) {
        data.col(i) = v.col(permutation[i]);
    }
    return data;
}

template <typename T, int DIM>
Eigen::Matrix<T, DIM, Eigen::Dynamic> copy_with_inverse_permutation(const Eigen::Matrix<T, DIM, Eigen::Dynamic> &v, const std::vector<size_t> &permutation)
{
    Eigen::Matrix<T, DIM, Eigen::Dynamic> data(DIM, v.cols());
    for (size_t i = 0; i < v.cols(); i++) {
        data.col(permutation[i]) = v.col(i);
    }
    return data;
}

    //assumes: p[0] in (0,1) and p[1] in (-pi,pi)
    template<size_t DIM,long int POINTS>
    inline Eigen::Vector<double, DIM> sphericalToCart(const Eigen::Ref<const Eigen::Array<double, DIM, POINTS, Eigen::RowMajor> >& p) 
    {
        Eigen::Array<double, DIM,POINTS, Eigen::RowMajor> res(POINTS,p.cols());

        if constexpr(DIM==2) {
	    res.row(0)= p.row(0)*Eigen::cos(p.row(1));
	    res.row(1)= p.row(0)*Eigen::sin(p.row(1));
        }else {
            assert(DIM==3);
	    res.row(0)=p.row(0)*Eigen::cos(p.row(2))*Eigen::sin(p.row(1));
	    res.row(1)=p.row(0)*Eigen::sin(p.row(2))*Eigen::sin(p.row(1));
	    res.row(2)=p.row(0)*Eigen::cos(p.row(2));	    
        }

        return  res;
    }

    

    template<size_t DIM,typename PointVector>
    inline typename PointVector::PlainObject interpToCart(const Eigen::ArrayBase<PointVector>& p, const Eigen::Vector<double, DIM> &xc, double H)
    {
	typename PointVector::PlainObject res(p.rows(),p.cols());

        if constexpr(DIM==2) {
	    res.row(0)= xc[0]+(H/p.row(0))*Eigen::cos(p.row(1));
	    res.row(1)= xc[1]+(H/p.row(0))*Eigen::sin(p.row(1));
        }else {
            assert(DIM==3);
	    res.row(0)=xc[0]+(H/p.row(0))*Eigen::cos(p.row(2))*Eigen::sin(p.row(1));
	    res.row(1)=xc[1]+(H/p.row(0))*Eigen::sin(p.row(2))*Eigen::sin(p.row(1));
	    res.row(2)=xc[2]+(H/p.row(0))*Eigen::cos(p.row(1));	    
        }

        return  res;
    }



    template<size_t DIM>
    inline Eigen::Vector<double, DIM> cartToSpherical(const Eigen::Ref<const Eigen::Vector<double, DIM> >& p) 
    {
	Eigen::Vector<double,DIM> xp = p ;


        if constexpr (DIM==2) {
	    const double r = xp.norm();

	    const long double theta = std::atan2( (long double) xp[1], (long double) xp[0]);

            Eigen::Vector<double, DIM> res;
            res[0] = r;
            res[1] = theta ;

            //assert(-1.0001 <= res[0] && res[0] <= 1.0001);
            //assert(-1 <= res[1] && res[1] <= 1);

            return  res;

        }else{
            assert(DIM==3);
            const double phi = std::atan2(xp[1], xp[0]);
            const double a=(xp[0]*xp[0]+xp[1]*xp[1]);
            const double theta= std::atan2(sqrt(a),xp[2]);
            const double r= sqrt(a+xp[2]*xp[2]);

            Eigen::Vector<double, DIM> res;
            res[0] = r;
            res[1] = theta;
            res[2] = phi;

            return  res;

        }
    }

    template<size_t DIM>
    inline Eigen::Vector<double, DIM> cartToInterp(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H) 
    {
	auto ps=cartToSpherical<DIM>(p-xc);
	ps[0]=H/ps[0];
	

	return ps;
    }
    
};

#endif
