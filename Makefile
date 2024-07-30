all:
	g++ -O2 -g -ldl -gdwarf-3  -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf test_ifgf.cpp -I/usr/include/eigen3/ -ltbb
	g++ -O2 -g -ldl -gdwarf-3  -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf_laplace test_ifgf_laplace.cpp -I/usr/include/eigen3/ -ltbb

test_cheb: test_chebinterp.cpp chebinterp.hpp
	g++  -O3 -g -march=native -std=c++20 -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
