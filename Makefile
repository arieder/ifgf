CPPFLAGS=-Ofast -g    -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20   -I/usr/include/eigen3/ -flto=auto

DEPS=chebinterp.hpp



all: 
	g++ $(CPPFLAGS) -o test_ifgf  test_ifgf.cpp  -ltbb
#	g++ -O2 -g -ldl -gdwarf-3  -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf_laplace test_ifgf_laplace.cpp -I/usr/include/eigen3/ -ltbb

#chebinterp.o: chebinterp.cpp chebinterp.hpp  boundingbox.hpp
#	g++ $(CPPFLAGS) -c chebinterp.cpp 


test_cheb: test_chebinterp.cpp chebinterp.hpp
	g++  -O3 -g -march=native -std=c++20 -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
