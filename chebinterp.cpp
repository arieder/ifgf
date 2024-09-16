#include <Eigen/Dense>
#include "chebinterp.hpp"





template <typename T, unsigned int DIM, unsigned int DIMOUT>
void ChebychevInterpolation::fast_evaluate_tp(
				 const Eigen::Ref<const Eigen::Array<double, DIM-1, Eigen::Dynamic> >
				  &points,
                                  const Eigen::Ref<const Eigen::Array<double, 1, Eigen::Dynamic> >
				  &points2,
                                  int axis,
				  const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
				  &interp_values,
				  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
				  const Eigen::Vector<int, DIM>& ns,
				 BoundingBox<DIM> box )
    {	


        dest.fill(0);
        Eigen::Array<double,DIM-1,Eigen::Dynamic> points_t(DIM-1,points.cols());
        Eigen::Array<double,1,Eigen::Dynamic> points2_t(1,points2.cols());


        const auto a=0.5*(box.max()-box.min()).array();
        const auto b=0.5*(box.max()+box.min()).array();
        
        if(axis==2) {
            if(!box.isNull()) {
                points_t.array()=(points.array().colwise()-b.topRows(2)).colwise()/a.topRows(2);
                points2_t.array()=(points2.array()-b(2))/a(2);
            }else {
                points_t=points;
                points2_t=points2;
            }
        }

	if(axis==0) {
            if(!box.isNull()) {
                points_t.array()=(points.array().colwise()-b.bottomRows(2)).colwise()/a.bottomRows(2);
                points2_t.array()=(points2.array()-b(0))/a(0);

            }else {
                points_t=points;
                points2_t=points2;
            }
        }


        size_t n_points = points_t.cols();
	size_t stride=n_points;
	size_t n_values = ns.prod()/ns[axis];
	size_t n_y = points2_t.cols();


	Eigen::Array<T, Eigen::Dynamic, 1> M(ns.prod());

	//std::cout<<"building m"<<DIM<<std::endl;
	if(axis==2) {
	    for(size_t idx=0;idx<ns[2];idx++) {
		__eval<T, DIM-1, 5>(points_t,interp_values.middleRows(idx*n_values,n_values),
                                    ns.template head<DIM-1>(), M.middleRows(idx*n_points,n_points).array(), 0, n_points);
	    }

	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> b1(n_points,DIMOUT);
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> b2(n_points,DIMOUT);
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> tmp(n_points,DIMOUT);


	    for(size_t sigma=0;sigma<n_y;sigma++) {		
		b1=2.*points2_t[sigma]*M.middleRows((ns[2]-1)*n_points,n_points)+M.middleRows((ns[2]-2)*n_points,n_points);
		b2=(M.middleRows((ns[2]-1)*n_points,n_points));

		for(size_t j=ns[2]-3;j>0;j--) {
		    tmp=(2.*((b1)*points2_t[sigma])-(b2))+M.middleRows(j*n_points,n_points);

		    b2=b1;
		    b1=tmp;
		}
	    
		dest.middleRows(sigma*n_points,n_points)=(b1*points2_t[sigma]-b2)+M.middleRows(0,n_points);
	    }
	}else if(axis==0) {
	    //std::cout<<"val="<<interp_values.matrix().norm()<<std::endl;

	    
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> tmp(n_y,DIMOUT);
	    for(size_t idx=0;idx<n_values;idx++) {
		tmp.fill(0);
		__eval<T, 1, 5>(points2_t,interp_values.middleRows(idx*ns[0],ns[0]),
				ns.template head<1>(), tmp, 0, n_y);

		for(size_t sigma=0;sigma<n_y;sigma++) {
		    M.row(sigma*n_values+idx)=tmp[sigma];
		}
	    }


	    tmp.resize(n_points);	                
	    for(size_t l=0;l<points2_t.size();l++){
		//Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, 0, Eigen::Stride<Eigen::Dynamic,1> > B
		//    (dest.data()+l, n_points, Eigen::Stride<Eigen::Dynamic,1>(n_points, 1));                   

		tmp.fill(0);
		__eval<T, DIM-1, 5>(points_t,M.segment(l*n_values,n_values),
				    ns.template tail(DIM-1), tmp, 0, n_points);

		for(size_t idx=0;idx<n_points;idx++) {		   
		    dest.row(idx*n_y+l)=tmp[idx];		    
		}
	    }
	}else{
	    assert(false); //not implemented
	}
            
        /*for(size_t l=0;l<points2.size();l++){
            if(axis==2) {                    
                __eval<T, DIM-1, 5>(points_t,interp_values.middleRows(l*n_values,n_values),
                                    ns.template head<DIM-1>(), dest.middleRows(l*n_points,n_points).array(), 0, n_points);
            }else if(axis==0) {
		Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>, 0, Eigen::Stride<Eigen::Dynamic,1> > A
		    (interp_values.data()+l, n_values, Eigen::Stride<Eigen::Dynamic,1>(n_values, 1));


		Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, 0, Eigen::Stride<Eigen::Dynamic,1> > B
		    (dest.data()+l, n_points, Eigen::Stride<Eigen::Dynamic,1>(n_points, 1));                   

                __eval<T, DIM-1, 5>(points_t,A,
                                    ns.template tail(DIM-1), B, 0, n_points);
            }
	    }*/
}


template <typename T, unsigned int DIM, unsigned int DIMOUT>
inline void  tp_evaluate_int(
			     const Eigen::Array<double, Eigen::Dynamic,1> * points,					 
			     const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
			     &interp_values,
			     Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
			     const Eigen::Vector<int, DIM>& ns )
{	

    assert(interp_values.size()==ns.prod());

    
    if constexpr (DIM==1) {	
	//std::cout<<"a"<<points[0].size()<<" "<<dest.rows()<<std::endl;
	//assert(points[0].size()==dest.rows());
	ChebychevInterpolation::parallel_evaluate<T,1,DIMOUT>(points[0].transpose(),interp_values,dest,ns);
    }else {	
	const size_t Ny=points[DIM-1].size();
	size_t Np=1; 
	for(int i=0;i<DIM-1;i++) {
	    Np*=points[i].size();
	}
	//std::cout<<"Np"<<Np<<std::endl;
	Eigen::Array<T, Eigen::Dynamic, 1> M(ns[DIM-1]*Np);

	const size_t n_values=ns.prod()/ns[DIM-1];

	for(size_t idx=0;idx<ns[DIM-1];idx++) {
	    tp_evaluate_int<T,DIM-1, DIMOUT>(points,interp_values.middleRows(idx*n_values,n_values),
					     M.middleRows(idx*Np,Np).array(),ns.template head<DIM-1>());
	}
	
	Eigen::Array<T, Eigen::Dynamic, DIMOUT> b1(Np,DIMOUT);
	Eigen::Array<T, Eigen::Dynamic, DIMOUT> b2(Np,DIMOUT);
	Eigen::Array<T, Eigen::Dynamic, DIMOUT> tmp(Np,DIMOUT);

	
	for(size_t sigma=0;sigma<Ny;sigma++) {
	    if(ns[DIM-1]<=2) {
		if(ns[DIM-1]==0) {
		    b1=M.middleRows(0*Np,Np);		   
		}else{
		    b1=points[DIM-1][sigma]*M.middleRows(1*Np,Np)+M.middleRows(0*Np,Np);		   
		}
		dest.middleRows(sigma*Np,Np)=b1;
	    }else {
		b1=2.*points[DIM-1][sigma]*M.middleRows((ns[DIM-1]-1)*Np,Np)+M.middleRows((ns[DIM-1]-2)*Np,Np);
		b2=(M.middleRows((ns[DIM-1]-1)*Np,Np));

		for(size_t j=ns[DIM-1]-3;j>0;j--) {
		    tmp=(2.*((b1)*points[DIM-1][sigma])-(b2))+M.middleRows(j*Np,Np);
		
		    b2=b1;
		    b1=tmp;
		}
	    
	    
		dest.middleRows(sigma*Np,Np)=(b1*points[DIM-1][sigma]-b2)+M.middleRows(0,Np);
	    }
	}
    }
	
}


template <typename T, unsigned int DIM, unsigned int DIMOUT>
void ChebychevInterpolation::tp_evaluate(
					 const std::array<Eigen::Array<double, Eigen::Dynamic,1> , DIM >
		 &points,					 
		 const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
		 &interp_values,
		 Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
		 const Eigen::Vector<int, DIM>& ns )
{
    tp_evaluate_int<T,DIM,DIMOUT>(points.data(), interp_values,dest,ns);
}


template
void ChebychevInterpolation::tp_evaluate<std::complex<double>, 3,1>(
								    const std::array<Eigen::Array<double,  Eigen::Dynamic,1> , 3 >
								    &points,					 
								    const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> >
								    &interp_values,
								    Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > dest,
								    const Eigen::Vector<int, 3>& ns );
    




template <typename T, unsigned int DIM, unsigned int DIMOUT>
void ChebychevInterpolation::parallel_evaluate(
					       const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >
					       &points,
					       const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
					       &interp_values,
					       Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
					       const Eigen::Vector<int, DIM>& ns,
					       BoundingBox<DIM> box)
    {
	
	Eigen::Array<double,DIM,Eigen::Dynamic> points0(DIM,points.cols());

	const auto a=0.5*(box.max()-box.min()).array();
	const auto b=0.5*(box.max()+box.min()).array();

	if(!box.isNull()) {
	    points0.array()=(points.array().colwise()-b).colwise()/a;
	}else {
	    points0=points;
	}

	//std::cout<<"ev"<<nodes[DIM-1]<<std::endl;

	//dest.resize(DIMOUT,points.cols());
        
	//template <typename T, int N_POINTS_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT, typename Derived1, typename Derived2, int N_AT_COMPILE_TIME, int... OTHER_NS>

#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem ifgf cheb::eval");

	ngcore::RegionTimer reg(t);
	t.AddFlops (ns.prod()*points.cols()*DIMOUT);
#endif


	//for(int i=0;i<points.cols();)
	//{
	size_t n_points = points.cols();
	//We do packages of size 4, 2, 1
	__eval<T, DIM, 5>(points0, interp_values, ns, dest, 0, n_points);
	    //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
	    //assert(i == r.end());
	 //}
	    

    }





// do some explicit initiations
template
const Eigen::Array<double, 3, Eigen::Dynamic> &ChebychevInterpolation::chebnodesNdd<double,3>( const Eigen::Ref< const Eigen::Vector<int, 3> >& ns);



//double
template struct ChebychevInterpolation::InterpolationData<double, 3, 1>;

template
struct ChebychevInterpolation::InterpolationData<double,3,3>;


template 
void ChebychevInterpolation::parallel_evaluate<double,3,1>(
			      const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic> >
			      &points,
			      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1> >
			      &interp_values,
			      Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> > dest,
			      const Eigen::Vector<int, 3>& ns,
			      BoundingBox<3> box = BoundingBox<3>());




template 
void ChebychevInterpolation::fast_evaluate_tp<double,3 ,1>(
				 const Eigen::Ref<const Eigen::Array<double, 2, Eigen::Dynamic> >
				  &points,
                                  const Eigen::Ref<const Eigen::Array<double, 1, Eigen::Dynamic> >
				  &points2,
                                  int axis,
				  const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1> >
				  &interp_values,
				  Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> > dest,
				  const Eigen::Vector<int, 3>& ns,
				 BoundingBox<3> box = BoundingBox<3>());





template
void ChebychevInterpolation::chebtransform<double,3>(const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1> >& src,
						     Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1> > dest,
						     const Eigen::Ref<const Eigen::Vector<int,3> >& ns
						     );
    

//complex

template struct ChebychevInterpolation::InterpolationData<std::complex<double>, 3, 1>;

template
struct ChebychevInterpolation::InterpolationData<std::complex<double>,3,3>;


template 
void ChebychevInterpolation::parallel_evaluate<std::complex<double>,3,1>(
			      const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic> >
			      &points,
			      const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> >
			      &interp_values,
			      Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > dest,
			      const Eigen::Vector<int, 3>& ns,
			      BoundingBox<3> box = BoundingBox<3>());




template 
void ChebychevInterpolation::fast_evaluate_tp<std::complex<double>,3 ,1>(
				 const Eigen::Ref<const Eigen::Array<double, 2, Eigen::Dynamic> >
				  &points,
                                  const Eigen::Ref<const Eigen::Array<double, 1, Eigen::Dynamic> >
				  &points2,
                                  int axis,
				 const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> >
				  &interp_values,
				 Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > dest,
				  const Eigen::Vector<int, 3>& ns,
				 BoundingBox<3> box = BoundingBox<3>());



template   
void ChebychevInterpolation::chebtransform<std::complex<double>,3>(const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> >& src,
								   Eigen::Ref<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > dest,
								   const Eigen::Ref<const Eigen::Vector<int,3> >& ns
								   );
    
