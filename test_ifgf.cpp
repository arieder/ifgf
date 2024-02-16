#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "ifgfoperator.hpp"
#include "octree.hpp"


const std::complex<double>  k= std::complex<double>(10,10);

std::complex<double> kernel(Eigen::Vector2d x,Eigen::Vector2d y)
{
    double d=(x-y).norm();
    
    return d==0 ? 0 : (1/(4*M_PI))*exp(-k*d)/d;
}



class MyIfgfOperator : public IfgfOperator<std::complex<double> ,2,MyIfgfOperator>
{
public:
    MyIfgfOperator(int leafSize): IfgfOperator(leafSize)
    {
    }

    typedef std::complex<double > T ;

    template<typename TX>
    inline T kernelFunction(TX x) const
    {
	double d=x.norm();
	return (d==0) ? 0 : (1/(4*M_PI))*exp(-k*d)/d;
    }


    

    template<typename TX>
    inline T CF(TX x) const
    {
	double d=x.norm();
	return exp(-k*d)/(4*M_PI*d);
    }


    template<typename TX,typename TY>
    inline T transfer_factor(TX x, TY xc, double H, TY pxc, double pH) const
    {
	/*yc  = IFGF.center(Y)
	  yp  = IFGF.center(IFGF.parent(Y))
	  d   = norm(x-yc)
	  dp  = norm(x-yp)
	  exp(im*K.k*(d-dp))*dp/d
	*/
	double d=(x-xc).norm();
	double dp=(x-pxc).norm();
	
	return exp(-k*(d-dp))*(dp/d);
    }

    
    void evaluateKernel(const Eigen::Ref<const PointArray> & x, const Eigen::Ref<const PointArray> & y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > & w,
			Eigen::Ref<Eigen::Vector<T,Eigen::Dynamic> >  result) const
    {
	assert(result.size()==y.cols());
	assert(w.size()==x.cols());
	
	for(int i=0;i<x.cols();i++)
	{	    
	    for(int j=0;j<y.cols();j++)
	    {
		result[j]+=w[i]*kernelFunction(x.col(i)-y.col(j));
	    }
	}
    }

    

    Eigen::Vector<T,Eigen::Dynamic>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> & x, const Eigen::Ref<const PointArray> & y,
							    const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> > & weights,
							    const Eigen::Vector<double,2> xc, double H) const
    {
	
	Eigen::Vector<T,Eigen::Dynamic> result(y.cols());
	result.fill(0);
	for(int j=0;j<y.cols();j++) {
	    double dc=(y.col(j)-xc).norm();
	    for(int i=0;i<x.cols();i++) {
		double d=(x.col(i)-y.col(j)).norm();
		result[j]+=weights[i]*
		    exp(-k*(d-dc))*(dc)/d;
		    //kernelFunction(x.col(i)-y.col(j))*inv_CF(y.col(j)-xc);
	    }
	}
	return result;
    }

    inline unsigned int orderForBox(double H, unsigned int baseOrder)
    {
	const int order = baseOrder+4*std::max(round( log(abs(k)*H)/log(2)),0.0);

	return order;
    }


};

#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
int main()    
{
    const int N=100000;
    typedef Eigen::Matrix<double, 2,Eigen::Dynamic> PointArray ;

    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism, 		1);
    //oneapi::tbb::task_arena arena(1);
    
    MyIfgfOperator op(10);

    PointArray srcs=(PointArray::Random(2,N).array());
    PointArray targets=(PointArray::Random(2,N).array()+1);

    op.init(srcs,targets);  

    
    Eigen::Vector<std::complex<double>,Eigen::Dynamic> weights(N);
    weights=Eigen::VectorXd::Random(N);

    Eigen::Vector<std::complex<double>,Eigen::Dynamic> result=op.mult(weights);
    std::cout<<"done multiplying"<<std::endl;

    srand((unsigned) time(NULL));
    double maxE=0;
    for(int j=0;j<1000;j++)
    {
	std::complex<double> val=0;
	int index=rand() % targets.cols();
	//std::cout<<"idx"<<index<<std::endl;
	for(int i=0;i<srcs.cols();i++)
	{
	    val+=weights[i]*kernel(srcs.col(i),targets.col(index));
	}

	double e=std::abs(val-result[index]);
	maxE=std::max(e,maxE);
	//std::cout<<"e="<<e<<" val="<<val<<" vs" <<result[index]<<std::endl;
    }


    std::cout<<"summary: e="<<maxE<<std::endl;

    
       

	

}
