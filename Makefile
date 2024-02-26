all:
	g++ -O3 -DNDEBUG -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf test_ifgf.cpp -I/usr/include/eigen3/ -ltbb

test_cheb: test_chebinterp.cpp chebinterp.hpp
	g++  -O3 -march=native -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
