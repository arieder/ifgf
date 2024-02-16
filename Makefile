all:
	g++  -O3 -march=native -DNDEBUG -ffast-math -std=c++20 -o test_ifgf test_ifgf.cpp -I/usr/include/eigen3/ -ltbb

test_cheb: test_chebinterp.cpp chebinterp.hpp
	g++  -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
