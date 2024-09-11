CPPFLAGS=-O2  -g   -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20   -I/usr/include/eigen3/

DEPS=chebinterp.hpp



all: chebinterp.o test_ifgf.o
	g++ $(CPPFLAGS) -o test_ifgf chebinterp.o test_ifgf.o  -ltbb
#	g++ -O2 -g -ldl -gdwarf-3  -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf_laplace test_ifgf_laplace.cpp -I/usr/include/eigen3/ -ltbb

chebinterp.o: chebinterp.cpp chebinterp.hpp  boundingbox.hpp
	g++ $(CPPFLAGS) -c chebinterp.cpp 

test_ifgf.o: chebinterp.o test_ifgf.cpp ifgfoperator.hpp cone_domain.hpp boundingbox.hpp octree.hpp helmholtz_ifgf.hpp
	g++ $(CPPFLAGS) -c test_ifgf.cpp 

test_cheb: test_chebinterp.cpp chebinterp.hpp
	g++  -O3 -g -march=native -std=c++20 -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
