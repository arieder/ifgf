#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <numeric>

#include <algorithm>

namespace Util{
    template <class ExecutionPolicy, typename RandomIt, class Compare>
    auto sort_with_permutation(ExecutionPolicy&& policy, RandomIt cbegin, RandomIt cend, Compare comp) {
	auto len = std::distance(cbegin, cend);
	std::vector<size_t> perm(len);
	std::iota(perm.begin(), perm.end(), 0U);
	std::sort(policy, perm.begin(), perm.end(),
		  [&](const size_t& a, const size_t& b)
		  {
		      return comp(*(cbegin+a), *(cbegin+b));
		  });
	return perm;
    }

    template <typename T, int DIM>
    Eigen::Matrix<T,DIM,Eigen::Dynamic> copy_with_permutation(const Eigen::Matrix<T, DIM, Eigen::Dynamic> & v, const std::vector<size_t> & permutation )
    {	
	Eigen::Matrix<T,DIM,Eigen::Dynamic> data(DIM,v.cols());
	for(size_t i=0;i<v.cols();i++) {
	    data.col(i)=v.col(permutation[i]);
	}
	return data;
    }


    template <typename T, int DIM>
    Eigen::Matrix<T,DIM,Eigen::Dynamic> copy_with_inverse_permutation(const Eigen::Matrix<T, DIM, Eigen::Dynamic> & v, const std::vector<size_t> & permutation )
    {	
	Eigen::Matrix<T,DIM,Eigen::Dynamic> data(DIM,v.cols());
	for(size_t i=0;i<v.cols();i++) {
	    data.col(permutation[i])=v.col(i);
	}
	return data;
    }


};

#endif
