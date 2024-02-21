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
    template<size_t DIM>
    inline Eigen::Vector<double, DIM> sphericalToCart(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H) 
    {
        const double theta = p[1];

        Eigen::Vector<double, DIM> res = xc;

        const double r = p[0];
        if constexpr(DIM==2) {
            res[0] += r * std::cos(theta);
            res[1] += r * std::sin(theta);
        }else {
            assert(DIM==3);
            const double phi=p[2];
            res[0] += r * cos(phi)*sin(theta);
            res[1] += r * sin(phi)*sin(theta);
            res[2] += r * cos(theta);
        }

        return  res;
    }

    template<size_t DIM>
    inline Eigen::Vector<double, DIM> interpToCart(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H)
    {
	const double  r=H/(p[0]);
	p[0]=r;
	auto px=sphericalToCart<DIM>(p,xc,H);
	
	return px;
    }



    template<size_t DIM>
    inline Eigen::Vector<double, DIM> cartToSpherical(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H) 
    {
	Eigen::Vector<double,DIM> xp = p - xc;


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
	auto ps=cartToSpherical<DIM>(p,xc,H);
	ps[0]=H/ps[0];

	//auto ps2=cartToSpherical<DIM>(ps,xc,H);

	return ps;
    }
    
};

#endif
