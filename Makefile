all:
	g++  -O3 -std=c++20 -o test_ifgf test_ifgf.cpp -I/usr/include/eigen3/ -ltbb

test_cheb:
	g++ -O3 -o test_cheb test_chebinterp.cpp -I/usr/include/eigen3/ -ltbb
