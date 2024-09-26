#ifndef __IFGFOPERATOR_HPP_
#define __IFGFOPERATOR_HPP_

#include "config.hpp"

#include <Eigen/Dense>
#include <tbb/queuing_mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/queuing_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>


#include "boundingbox.hpp"
#include "cone_domain.hpp"
#include "octree.hpp"
#include "chebinterp.hpp"

//#include <fstream>
#include <iostream>

#include <memory>

template<typename T, unsigned int DIM, unsigned int DIMOUT, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Array<double, DIM, Eigen::Dynamic> PointArray;     //, Eigen::RowMajor?

    enum RefinementType { RefineH, RefineP};

    IfgfOperator(long int maxLeafSize = -1, size_t order=5, size_t n_elements=1, double tolerance = -1)
    {
	assert(n_elements>0);
	if constexpr (DIM==3) {
	    m_base_n_elements[0]=1;
	    m_base_n_elements[1]=2;
	    m_base_n_elements[2]=2;
	}else {
	    m_base_n_elements[0]=1;
	    m_base_n_elements[1]=2;
	}
	m_base_n_elements*=n_elements;	

	std::cout<<"creating new ifgf operator. n_leaf="<<maxLeafSize<<" order= "<<order<<" n_elements="<<n_elements<<std::endl;
        m_src_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);
	m_target_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);
	m_baseOrder=order;
	m_tolerance=tolerance;


    }

    ~IfgfOperator()
    {

    }

    const Octree<T,DIM>& src_octree() const {
	return *m_src_octree;
    }



    void init(const PointArray &srcs, const PointArray targets)
    {
        m_src_octree->build(srcs);
	m_target_octree->build(targets);
	
	m_src_octree->buildInteractionList(*m_target_octree);


	static_cast<Derived *>(this)->onOctreeReady();
        //m_src_octree->sanitize();

        m_numTargets = targets.cols();
        m_numSrcs = srcs.cols();

	if(m_tolerance>0) {
	    if(m_tolerance> 1e-4) 
		m_baseOrder=4;
	    else if(m_tolerance > 1e-10)
		m_baseOrder=8;
	    else
		m_baseOrder=16;

	    m_base_n_elements*=estimateRefinement(m_tolerance,RefineH);
	    //m_baseOrder=estimateOrder(m_tolerance);
	}


	std::cout<<"calculating interp range"<<std::endl;
	m_src_octree->calculateInterpolationRange([this](double H,int step){return static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);},
						  [this](double H, int step){return static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements,step);},*m_target_octree);

	std::cout<<"done initializing"<<std::endl;
    }


    
    int estimateRefinement(double tol,RefinementType refine)
    {
	std::cout<<"estimating the order needed to achieve "<<tol<< "using "<< (refine==RefineH ? "h":"p")<<"-refinement"<<std::endl;
	//use n boxes randomly to estimate the interpolation error
	const size_t level=m_src_octree->levels()-1;
	const size_t Nboxes= m_src_octree->numBoxes(level);
	const size_t sampleBoxes=10;
	const size_t stride= Nboxes/sampleBoxes;

	std::cout<<"working on level"<<level<<" "<<m_src_octree->numBoxes(level)<<std::endl;

	auto ref=tbb::parallel_reduce(
					tbb::blocked_range<int>(0,sampleBoxes),
					1,
					[&](tbb::blocked_range<int> r, int order) {
					    for(size_t i=r.begin();i<r.end();i++)
					    {
						const size_t boxId=i*stride;
						order=std::max(order, estimateRefinementOnBox(tol,level,boxId,refine));
					    }
					    return order;
					},[](int a, int b){return std::max(a,b);});

	std::cout<<"using refinemnt="<<ref<<std::endl;;
	return ref;
    }

    int estimateRefinementOnBox(double tol,size_t level,size_t id, RefinementType refine)
    {
	/*const int maxR=refine== RefineH ? 4 : 15;
	
	BoundingBox bbox = m_src_octree->bbox(level, id);
        auto center = bbox.center();
        double H = bbox.sideLength();
        IndexRange srcs = m_src_octree->points(level, id);
        const size_t nS = srcs.second - srcs.first;
	
	double smax=sqrt(DIM)/DIM;
	//if the sources and targets are well-separated we don't have to cover the near field 
	const double dist=m_src_octree->bbox(0,0).exteriorDistance(m_target_octree->bbox(0,0));
	if(dist >0) {
	    smax=std::min(smax, H/dist);
	}
	const double smin=H/(m_target_octree->bbox(0,0).distanceToBoundary(center));

	
	BoundingBox<DIM> int_box;
	int_box.min()(0)=smin;
	int_box.max()(0)=smax;
	
	const size_t n_samples=150;
	//now scale to (smin,smax) x (0,PI) x (-M_PI,M_PI) (in 3d)
	if constexpr(DIM==2) {	    
	    int_box.min()(1)=-M_PI;
	    int_box.max()(1)=M_PI;
	}else{
	    int_box.min()(1)=0;
	    int_box.max()(1)=M_PI;
	    
	    int_box.min()(2)=-M_PI;
	    int_box.max()(2)=M_PI;
	}

	int base=0;
	double error=std::numeric_limits<double>::max();
	int new_p=1;
	int new_n_els=0;

	while(error > tol && base<maxR) {
	    ++base;
	    new_p=refine==RefineP ? m_baseOrder+base: m_baseOrder;
	    new_n_els=refine==RefineH ?  pow(2,base-1) : 1;
	    
		
	    const auto order = static_cast<Derived *>(this)->orderForBox(H, new_p );
	    //std::cout<<"trying order"<<order<<std::endl;
	    Eigen::Vector<size_t, DIM> n_els= static_cast<Derived *>(this)->elementsForBox(H, new_p, m_base_n_elements * new_n_els );

	    ConeDomain<DIM> grid(n_els,int_box);
	    PointArray chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
	    PointArray transformedNodes(DIM,chebNodes.cols());
		
	    const size_t stride=chebNodes.cols();
	    error=0;
	    for (int el =0;el<grid.n_elements();el++) {
		Eigen::Vector<T,Eigen::Dynamic> weights=Eigen::Vector<T,Eigen::Dynamic>::Random(nS);
		transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, center, H);

		auto data= static_cast<const Derived *>(this)->evaluateFactoredKernel
		    (m_src_octree->points(srcs), transformedNodes, weights, center, H,srcs);

		
		transformInterpToCart(grid.transform(el,samplePoints), transformedSample, center, H);

		auto exact= static_cast<const Derived *>(this)->evaluateFactoredKernel
		    (m_src_octree->points(srcs), transformedSample, weights, center, H,srcs);

		double norm=std::max(exact.matrix().norm(),1.);
		Eigen::Array<T,Eigen::Dynamic,DIMOUT> approx(samplePoints.cols(),DIMOUT);
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(samplePoints,data,approx,order);
		exact.matrix()-=approx.matrix();
		error=std::max(error,exact.cwiseAbs().maxCoeff()/norm);
	    }
	    //std::cout<<"error2="<<error<<" at "<<base<<std::endl;;
	    //error=sqrt(error)/grid.n_elements();
	    //std::cout<<"order="<<order<<" error="<<error<<std::endl;
	}

	return refine == RefineH ? new_n_els : new_p;

	*/

	return 1;
    }
    
    


    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	std::cout<<"mult "<<m_baseOrder<<std::endl;

	std::cout<<"permutation"<<std::endl;
        Eigen::Vector<T, Eigen::Dynamic> new_weights = Util::copy_with_permutation (weights, m_src_octree->permutation());
	
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(m_numTargets,DIMOUT);
        result.fill(0);
        int level = m_src_octree->levels() - 1;

	std::cout<<"boxes="<<m_src_octree->numBoxes(level)<<std::endl;
	const double hmin=m_src_octree->diameter()*std::pow(0.5,m_src_octree->levels());
	std::cout<<"base size"<<static_cast<Derived *>(this)->elementsForBox(hmin, m_baseOrder,this->m_base_n_elements).transpose()<<std::endl;
	std::cout<<"now go"<<std::endl;

	//std::vector<tbb::queuing_mutex> resultMutex(m_numTargets);

	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > local_result(result);
        tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > tmp_result;
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, 1 > > tmp_chebt;
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, 1 > > tmp_coordTrafo;
	tbb::enumerable_thread_specific<PointArray > transformedNodes;

#ifndef RECURSIVE_MULT
	std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > interpolationData(m_src_octree->numBoxes(level));
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > parentInterpolationData;
#endif

#ifdef CHECK_CONNECTIVITY
	std::cout<<"srcs="<<m_numSrcs<<" "<<m_numTargets<<std::endl;
	m_connectivity.resize(m_numSrcs, m_numTargets);
        m_connectivity.fill(0);
	std::cout<<"done"<<std::endl;
#endif
#ifdef  RECURSIVE_MULT
	int min_boxes=8;
        int recursive_level=0;
        for(;recursive_level<m_src_octree->levels();recursive_level++) {
            if(m_src_octree->numBoxes(recursive_level) > min_boxes) {
                break;
            }
        }
        //recursive_level=m_src_octree->levels()-1;

        level=recursive_level;

        std::cout<<"doing recursive at level"<<level<<std::endl;

        std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> >
            interpolationData(m_src_octree->numBoxes(level-1));
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> >
            parentInterpolationData;
       

	//prepare the int data
	Eigen::Vector<int, DIM> order;
	
	for(size_t pId=0;pId<m_src_octree->numBoxes(level-1);pId++) {
	    //if( !m_src_octree->hasFarTargetsIncludingAncestors(level-1,pId)) {
	    //	continue;
	    //}
	    BoundingBox bbox = m_src_octree->bbox(level - 1, pId);
	    double H = bbox.sideLength();
	    order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,1);	    
	    auto grid= m_src_octree->coneDomain(level-1,pId);		
	    interpolationData[pId].grid = grid;
	    interpolationData[pId].values.resize(grid.activeCones().size()*order.prod(),DIMOUT);            
	    //interpolationData[pId].values.fill(0);
	    interpolationData[pId].order = order;                        
        }


	
	tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numBoxes(level)),
        [&](tbb::blocked_range<size_t> r) {
            for(size_t id=r.begin();id<r.end();id++) {
                recursive_mult(id, level, new_weights,local_result.local(), interpolationData[m_src_octree->parentId(level, id)],
			       tmp_result.local(),  transformedNodes.local());                
            }});

	//now collect the different results
	result=local_result.combine([](auto a,auto b){return a+b;});
	
        
        /*for(size_t id=0; id < m_src_octree->numBoxes(level); id++)
          recursive_mult(id, level, new_weights,result, interpolationData[m_src_octree->parentId(level, id)],
	  tmp_result.local(), resultMutex, transformedNodes.local());                */
        
	level--;
        std::cout<<"now proceeding iteratively"<<level<<std::endl;
#endif
	tbb::queuing_mutex resultMutex;
		
        for (; level >= 0; --level) {
            std::cout << "level=" << level << " "<< m_src_octree->numBoxes(level)<< std::endl;
	    

            std::cout << "near field" <<std::endl;
	    
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numBoxes(level)),
	    [&](tbb::blocked_range<size_t> r) {
		for (size_t i = r.begin(); i < r.end(); i++) {
		    if(!m_src_octree->hasPoints(level,i))
			continue;
		    BoundingBox bbox = m_src_octree->bbox(level, i);
		    auto center = bbox.center();
		    double H = bbox.sideLength();
		    evaluateNearField(level,i, new_weights, result, tmp_result.local(),&resultMutex);
	    }});



	    //Get an exemplary bbox to determine the interpolation order
	    BoundingBox bbox = m_src_octree->bbox(level, 0);
	    double H0 = bbox.sideLength();
	    const auto order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,0);
	    const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
	    const auto high_order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,1);
	    const auto& ho_chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(high_order);

            const size_t stride=chebNodes.cols();

	    //make sure the factors for the chebtrafo are precomputed...
	    for(int d=0;d<DIM;d++) {
		ChebychevInterpolation::chebvals<double>(order[d]);
	    }


	    //there is no more far field or interpolation happening
	    if(level<=1) {
		break;
	    }




	    //prepare the interpolation data for all leaves
	    if(level==m_src_octree->levels()-1) {
		initInterpolationData(level,1, interpolationData);
	    }

	    std::cout<<"interpolate leaves"<<m_src_octree->numLeafCones(level)<<std::endl;

	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numLeafCones(level)),
		[&](tbb::blocked_range<size_t> r) {
#ifdef USE_NGSOLVE
		    static ngcore::Timer t("ngbem interpolate leaves");
		    ngcore::RegionTimer reg(t);
#endif

		for (size_t i = r.begin(); i < r.end(); i++) {
		    const ConeRef ref=m_src_octree->leafCone(level,i);
		    const size_t boxId=ref.boxId();

		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
			continue;
		    }

		    
		    assert(m_src_octree->isLeaf(level,boxId)==true);
		    BoundingBox bbox = m_src_octree->bbox(level, boxId);
		    auto center = bbox.center();
		    double H = bbox.sideLength();

		    auto grid=m_src_octree->coneDomain(level,boxId, 1);

		    IndexRange srcs=m_src_octree->points(level,boxId);
		    const size_t nS=srcs.second-srcs.first;

		    transformedNodes.local().resize(3,ho_chebNodes.cols());

		    const size_t stride=ho_chebNodes.cols();			
		    transformInterpToCart(grid.transform(ref.id(),ho_chebNodes), transformedNodes.local(), center, H);
		    interpolationData[boxId].values.middleRows(ref.memId()*stride,stride) =
			static_cast<const Derived *>(this)
			->evaluateFactoredKernel(m_src_octree->points(srcs),
						 transformedNodes.local(),
						 new_weights.segment(srcs.first, nS), center, H,srcs);			

		}});
	

	    //chebtrafo everything
	    std::cout<<"chebtrafo"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,1);
		size_t boxId=cone.boxId();
		if(!m_src_octree->hasPoints(level,boxId))
		    continue;

		
		if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
		    continue;
		}

		BoundingBox bbox = m_src_octree->bbox(level, boxId);
		auto center = bbox.center();
		double H = bbox.sideLength();

		const size_t stride=ho_chebNodes.cols();			
		//before we can use the interpolation data, we habe to run a chebychev transform on it

		tmp_chebt.local().resize(stride);

		ChebychevInterpolation::chebtransform<T,DIM>(interpolationData[boxId].values.middleRows(cone.memId()*stride,stride),tmp_chebt.local(),high_order);
		interpolationData[boxId].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();

	    }});


	    
	    //interpolation Data contains the values using the coarse- high-order interpolation scheme. Project to the low-order fine grid
	    //which is faster for point-evaluations. We use parentInteprolationData as a termpoary buffer.
	    initInterpolationData(level,0, parentInterpolationData);
	    std::cout<<"reinterpolating"<<std::endl;
	    
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)), //iterative over the few-high order cones to take full advantage of the TP structure
		[&](tbb::blocked_range<size_t> r) {
#ifdef USE_NGSOLVE
	    static ngcore::Timer t("ngbem reinterpolate");
	    ngcore::RegionTimer reg(t);
#endif
	    for (size_t i = r.begin(); i < r.end(); i++) {
		    //parent node
		    ConeRef hoCone=m_src_octree->activeCone(level,i,1);
		    size_t boxId=hoCone.boxId();
		    const auto& hoGrid= m_src_octree->coneDomain(level,boxId,1);
		    
		    if (!m_src_octree->hasPoints(level, boxId)) {
			continue;
		    }

		    //BoundingBox region=//hoGrid.region(hoCone.id());


		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
			continue;
		    }

		    BoundingBox bbox = m_src_octree->bbox(level, boxId);
		    auto center = bbox.center();
		    double H = bbox.sideLength();


		    coarseToFine(interpolationData[boxId], level, hoCone, tmp_result.local(), order, high_order,H,parentInterpolationData[boxId]);
	    }});

	    

	    // chebtrafo everything again now on the finer grid
	    std::cout<<"fine chebtrafo"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,0)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,0);
		size_t boxId=cone.boxId();
		if(!m_src_octree->hasPoints(level,boxId))
		    continue;
		
		//const size_t stride=chebNodes.cols();			
		//before we can use the interpolation data, we habe to run a chebychev transform on it

		tmp_chebt.local().resize(stride);		
		ChebychevInterpolation::chebtransform<T,DIM>(parentInterpolationData[boxId].values.middleRows(cone.memId()*stride,stride),tmp_chebt.local(),order);
		parentInterpolationData[boxId].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();
	    }});


	    std::cout<<"swapping"<<std::endl;
	    std::swap(interpolationData,parentInterpolationData);
	    parentInterpolationData.resize(0);

	    std::cout<<"evaluate far field"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_target_octree->numPoints()),
	        [&](tbb::blocked_range<size_t> r) {
		    tmp_result.local().resize(1,DIMOUT);
		for (size_t i = r.begin(); i < r.end(); i++) {
		    for( size_t boxId : m_src_octree->farfieldBoxes(level,i) ){
			BoundingBox bbox = m_src_octree->bbox(level, boxId);
			auto center = bbox.center();
			double H = bbox.sideLength();

#ifdef CHECK_CONNECTIVITY
			{
			    tbb::queuing_mutex::scoped_lock lock(m_conMutex);
			    IndexRange srcs=m_src_octree->points(level,boxId);
			    int q=i;
			    for (int k = srcs.first; k < srcs.second; k++) {
				/*if(m_connectivity(q,k)!=0) {
				  std::cout<<"farerror"<<q<<" "<<k<<std::endl;
				  }*/
				
				m_connectivity(k, q) += 1;
			    }
			}
			    
#endif

			//evaluate for the cousin targets using the interpolated data
			evaluateSingleFromInterp(interpolationData[boxId], m_target_octree->point(i), center, H,
					   tmp_result.local());

			result.row(i) += tmp_result.local();
		    }
		}});
	
	    /*#ifdef CHECK_CONNECTIVITY
            std::cout<<"connectivity"<<std::endl<<m_connectivity<<std::endl;
	    #endif*/


	    std::cout<<"propagating"<<std::endl;

            //Now transform the interpolation data to the parents
	    std::cout<<"propagating upward"<<std::endl;
	    initInterpolationData(level-1,1, parentInterpolationData);
#ifndef BE_FAST
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level-1,1)),
		[&](tbb::blocked_range<size_t> r) {
		    tmp_result.local().resize(ho_chebNodes.cols(),DIMOUT);
		    transformedNodes.local().resize(3,ho_chebNodes.cols());
#ifdef USE_NGSOLVE
	    static ngcore::Timer t("ngbem propagate up");
	    ngcore::RegionTimer reg(t);
#endif

		for (size_t i = r.begin(); i < r.end(); i++) {
		    //parent node
		    ConeRef parentCone=m_src_octree->activeCone(level-1,i,1);
                    size_t parentId = parentCone.boxId();
		    auto pGrid= m_src_octree->coneDomain(level-1,parentId,1);				    
                    BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
                    auto parent_center = parent_bbox.center();
                    double pH = parent_bbox.sideLength();

		    if (!m_src_octree->hasPoints(level-1, parentId)) {
			   continue;
		   }


		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level-1, parentId)){ //we dont need the interpolation info for those levels.
			continue;
		    } 

		    transformInterpToCart(pGrid.transform(parentCone.id(),ho_chebNodes), transformedNodes.local(), parent_center, pH);
		    const size_t stride=ho_chebNodes.cols();					

		    //std::cout<<"pc"<<parentCone.id()<<" "<<parentCone.memId()<<" "<<nterpolationData[parentId].values.size()<<std::endl;
		    parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride).fill(0);
		    for(size_t childBox : m_src_octree->childBoxes(level-1,parentId) ) {
			//current node
			BoundingBox bbox = m_src_octree->bbox(level, childBox);
			auto center = bbox.center();
			double H = bbox.sideLength();
			
			
			//tmp_result.local().fill(0);
			transferInterp(interpolationData[childBox], transformedNodes.local(), center, H, parent_center, pH, tmp_result.local());
						
			parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride)+=tmp_result.local();
		    }
		    		    
		    /*//now do the chebtrafo
		    tmp_chebt.local().resize(stride);
		    ChebychevInterpolation::chebtransform<T,DIM>(interpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride),tmp_chebt.local(),high_order);
		    parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride)=tmp_chebt.local();*/
	    }});
#else //BE_FAST


	    std::cout<<"rot1"<<std::endl;
	    initInterpolationData(level,2,parentInterpolationData);
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,2)),
	        [&](tbb::blocked_range<size_t> r) {
		    const size_t stride=ho_chebNodes.cols();					
		    tmp_result.local().resize(ho_chebNodes.cols(),DIMOUT);
		    tmp_coordTrafo.local().resize(ho_chebNodes.cols());
		    tmp_chebt.local().resize(ho_chebNodes.cols());
		    for (size_t i = r.begin(); i < r.end(); i++) {
			//current node
			ConeRef cone=m_src_octree->activeCone(level,i,2);
			size_t childBox=cone.boxId();
			BoundingBox bbox = m_src_octree->bbox(level, cone.boxId());
			auto center = bbox.center();
			double H = bbox.sideLength();

			size_t parentId = m_src_octree->parentId(level,cone.boxId());
			BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
			auto parent_center = parent_bbox.center();
			double pH = parent_bbox.sideLength();

			if (!m_src_octree->hasPoints(level-1, parentId)) {
			    continue;
			}
			
			if(level<2 || ! m_src_octree->parentHasFarTargets(level, cone.boxId())){ //we dont need the interpolation info for those levels.
			    continue;
			}

			//now reinterpolate to a polar coordinate system aligned with the parent box center via point-and shoot
			transferRotation(interpolationData[cone.boxId()], cone, parentInterpolationData[cone.boxId()].grid, center-parent_center, tmp_coordTrafo.local(),true, parentInterpolationData[childBox].order);

			//double n=tmp_coordTrafo.local().matrix().norm();
			//assert(n<1e5);
			
			ChebychevInterpolation::chebtransform<T,DIM>(tmp_coordTrafo.local(),tmp_chebt.local(),order);
			parentInterpolationData[cone.boxId()].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();			
		    }});
	    
	    std::cout<<"trans"<<std::endl;
	    initInterpolationData(level,3,interpolationData);
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,3)),
	        [&](tbb::blocked_range<size_t> r) {
		    const size_t stride=ho_chebNodes.cols();					
		    tmp_result.local().resize(ho_chebNodes.cols(),DIMOUT);
		    tmp_coordTrafo.local().resize(ho_chebNodes.cols());
		    tmp_chebt.local().resize(ho_chebNodes.cols());
		    for (size_t i = r.begin(); i < r.end(); i++) {
			//current node

			ConeRef cone=m_src_octree->activeCone(level,i,3);
			size_t childBox=cone.boxId();
			BoundingBox bbox = m_src_octree->bbox(level, cone.boxId());
			auto center = bbox.center();
			double H = bbox.sideLength();

			size_t parentId = m_src_octree->parentId(level,cone.boxId());
			BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
			auto parent_center = parent_bbox.center();
			double pH = parent_bbox.sideLength();

			if (!m_src_octree->hasPoints(level-1, parentId)) {
			    continue;
			}
			
			if(level<2 || ! m_src_octree->parentHasFarTargets(level,cone.boxId())){ //we dont need the interpolation info for those levels.
			    continue;
			}

			//now reinterpolate to a polar coordinate system aligned with the parent box center via point-and shoot
			//tmp_coordTrafo.local().fill(0);			
			transferTranslation(parentInterpolationData[cone.boxId()], cone, interpolationData[cone.boxId()].grid, center ,H, parent_center, pH, tmp_coordTrafo.local());

			//double n=tmp_coordTrafo.local().matrix().norm();
 			//std::cout<<"bla "<<n<<"  "<<tmp_interpolationData[cone.boxId()].values.matrix().norm()<<std::endl;
			
			//((assert(n<1e5);

			ChebychevInterpolation::chebtransform<T,DIM>(tmp_coordTrafo.local(),tmp_chebt.local(),order);
			interpolationData[cone.boxId()].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();			
		    }});
	    std::cout<<"rot2"<<std::endl;	    
	    initInterpolationData(level-1,1, parentInterpolationData);
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level-1,1)),[&](tbb::blocked_range<size_t> r) {
		const size_t stride=ho_chebNodes.cols();					
		tmp_result.local().resize(ho_chebNodes.cols(),DIMOUT);
		transformedNodes.local().resize(3,ho_chebNodes.cols());
		for (size_t i = r.begin(); i < r.end(); i++) {
		    //parent node
		    ConeRef parentCone=m_src_octree->activeCone(level-1,i,1);
                    size_t parentId = parentCone.boxId();
		    auto pGrid= m_src_octree->coneDomain(level-1,parentId,1);				    
                    BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
                    auto parent_center = parent_bbox.center();
                    double pH = parent_bbox.sideLength();

		    if (!m_src_octree->hasPoints(level-1, parentId)) {
			   continue;
		   }

		    if(level<2 || ! m_src_octree->hasFarTargetsIncludingAncestors(level-1, parentId)){ //we dont need the interpolation info for those levels.
		    	continue;
		    } 

		    transformInterpToCart(pGrid.transform(parentCone.id(),ho_chebNodes), transformedNodes.local(), parent_center, pH);

		    parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride).fill(0);
		    for(size_t childBox : m_src_octree->childBoxes(level-1,parentId) ) {
			//current node
			BoundingBox bbox = m_src_octree->bbox(level, childBox);
			auto center = bbox.center();
			double H = bbox.sideLength();				
			
			transferRotation(interpolationData[childBox], parentCone, pGrid, center-parent_center, tmp_result.local(),false, parentInterpolationData[childBox].order);
			static_cast<const Derived *>(this)->transfer_factor(transformedNodes.local(), center, H, parent_center, pH, tmp_result.local());
			parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride)+=tmp_result.local();
		    }
		}});




#endif


	    std::cout<<"swapping"<<std::endl;
            std::swap(interpolationData, parentInterpolationData);
            parentInterpolationData.resize(0);

        }

#ifdef CHECK_CONNECTIVITY
	std::cout<<"checking if ever src-dest pairing was hit"<<std::endl;
	if((m_connectivity.array() - 1).matrix().norm() > 1e-10) {
	    std::cout<<m_connectivity<<std::endl;
	}
        assert((m_connectivity.array() - 1).matrix().norm() < 1e-10);
#endif
        std::cout<<"done"<<std::endl;
        return Util::copy_with_inverse_permutation(result, m_target_octree->permutation());
    }


    void initInterpolationData(size_t level, size_t step, std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> >& i_data)
    {
	i_data.resize(m_src_octree->numBoxes(level));
	tbb::parallel_for(
                tbb::blocked_range<size_t>(0, i_data.size()),
                [&](tbb::blocked_range<size_t> r) {
                for (int id = r.begin(); id < r.end(); id++) {
		    BoundingBox bbox = m_src_octree->bbox(level , id);
		    //std::cout<<"bbox="<<bbox.min().transpose()<<" "<<bbox.max().transpose()<<std::endl;
		    auto center = bbox.center();
		    double H = bbox.sideLength();

		    auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);
		    
		    auto grid= m_src_octree->coneDomain(level,id,step);		

		    //std::cout<<step<<" actite="<<grid.activeCones().size()<<std::endl;
		    i_data[id].values.resize(grid.activeCones().size()*order.prod(),DIMOUT);
		    //i_data[id].values.fill(0);
		    i_data[id].order=order;
		    i_data[id].grid=grid;
	       }
	    });

    }

    void evaluateNearField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			   Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,
			   tbb::queuing_mutex* result_mutex=0
			   )
    {
#ifdef USE_NGSOLVE
      static ngcore::Timer t("ngbem eval Near Field");
      ngcore::RegionTimer reg(t);
#endif
      
	IndexRange srcs = m_src_octree->points(level, id);
	const size_t nS = srcs.second - srcs.first;

	if(nS==0) //skip empty boxes
	    return;

	BoundingBox bbox = m_src_octree->bbox(level, id);
	auto center = bbox.center();
	double H = bbox.sideLength();

	const auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder);
	const auto& chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order);
	
	std::vector<IndexRange> targetList = m_src_octree->nearTargets(level, id);

	for (const auto &targets : targetList) {
	    const size_t nT = targets.second - targets.first;
#ifdef CHECK_CONNECTIVITY
	    {
	    tbb::queuing_mutex::scoped_lock lock(m_conMutex);
	    for (int l = targets.first; l < targets.second; l++) {
		for (int k = srcs.first; k < srcs.second; k++) {
		    /*if(m_connectivity(k,l)!=0) {
			std::cout<<"near error"<<k<<" "<<l<<std::endl;
			}*/
		    m_connectivity(k, l) += 1;
		}
	    }
	    }
#endif

	    //std::cout<<"srcs="<<srcs.first<<" "<<srcs.second<<" "<<new_weights.size()<<std::endl;
	    //std::cout<<"targets="<<targets.first<<" "<<targets.second<<" "<<std::endl;

	    tmp_result.resize(nT,DIMOUT);
	    tmp_result.fill(0);

	    static_cast<Derived *>(this)->evaluateKernel(
							 m_src_octree->points(srcs),
							 m_target_octree->points(targets),
							 weights.segment(srcs.first, nS),
							 tmp_result,srcs);
		    
	    {
		if(result_mutex) {
		    tbb::queuing_mutex::scoped_lock lock(*result_mutex);
		    result.middleRows(targets.first, nT) += tmp_result;
		}else{
		    result.middleRows(targets.first, nT) += tmp_result;
		}
	    }
	    
	}	

    }

    void evaluateFarField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			  ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& interpolationData,
			  Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result			  
			 )
    {
#ifdef USE_NGSOLVE
      static ngcore::Timer t("ngbem eval Far Field");
      ngcore::RegionTimer reg(t);


#endif
      
	BoundingBox bbox = m_src_octree->bbox(level, id);
        auto center = bbox.center();
        double H = bbox.sideLength();
        
	//evaluate for the cousin targets using the interpolated data
	const std::vector<IndexRange> cousinTargets = m_src_octree->farTargets(level, id);
	for (unsigned int l = 0; l < cousinTargets.size(); l++) {
	    const size_t nT = cousinTargets[l].second - cousinTargets[l].first;
	    
#ifdef CHECK_CONNECTIVITY
	    IndexRange srcs=m_src_octree->points(level,id);
	    {
	    tbb::queuing_mutex::scoped_lock lock(m_conMutex);
	    for (int q = cousinTargets[l].first; q < cousinTargets[l].second; q++) {
		for (int k = srcs.first; k < srcs.second; k++) {
		    /*if(m_connectivity(q,k)!=0) {
			std::cout<<"farerror"<<q<<" "<<k<<std::endl;
			}*/

		    m_connectivity(k, q) += 1;
		}
	    }
	    }
#endif
	    tmp_result.resize(nT,DIMOUT);
			
	    evaluateFromInterp(interpolationData, m_target_octree->points(cousinTargets[l]), center, H,
			       tmp_result);
	    
	    {
		//tbb::queuing_mutex::scoped_lock lock(resultMutex[cousinTargets[l].first]);
		result.middleRows(cousinTargets[l].first, nT) += tmp_result;
	    }
	}  
    }



    
    inline void recursive_mult(long int id ,size_t level,
                               const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
                               Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
                               ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& parentData,
                               Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,                               
			       PointArray & transformedNodes
                               )
    {
        //std::cout<<"recursive mult"<<level<<" "<<id<<std::endl;
        
        if(id<0 || !m_src_octree->hasPoints(level,id))
            return;

        ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> storage;


        BoundingBox bbox = m_src_octree->bbox(level, id);
        auto center = bbox.center();
        double H = bbox.sideLength();
        IndexRange srcs = m_src_octree->points(level, id);
        const size_t nS = srcs.second - srcs.first;

        const auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder);
	const auto high_order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder, 1);


	auto fine_N=static_cast<Derived *>(this)->elementsForBox(H, m_baseOrder,m_base_n_elements,0);
       
        auto ho_grid=m_src_octree->coneDomain(level,id,1);
	auto lo_grid=m_src_octree->coneDomain(level,id,0);
	
	
        //std::cout<<"grid="<<grid.domain().min().transpose()<<std::endl;
        storage.order = high_order;
        storage.grid = ho_grid;
	const auto& ho_chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(high_order);
        storage.values.resize(ho_grid.activeCones().size()*ho_chebNodes.cols(),DIMOUT);

	transformedNodes.resize(DIM,ho_chebNodes.cols());



	//evaluate the near field exactly
	evaluateNearField(level,id,weights, result, tmp_result);	
                
        //if we are at the leaf-level compute the interpolation data by actually interpolating the true function
        if(m_src_octree->isLeaf(level,id)) {                                
            //std::cout<<"interp"<<level<<storage.values.size()<<std::endl;
            const size_t stride=ho_chebNodes.cols();
            for (int memId =0;memId<ho_grid.activeCones().size();memId++) {
                //std::cout<<"mem"<<memId;
                const size_t el=ho_grid.activeCones()[memId];
		
                transformInterpToCart(ho_grid.transform(el,ho_chebNodes), transformedNodes, center, H);
		storage.values.middleRows(memId*stride,stride)=
                    static_cast<const Derived *>(this)->evaluateFactoredKernel
                    (m_src_octree->points(srcs), transformedNodes, weights.segment(srcs.first, nS), center, H,srcs);
            }
        }else //generate the interpolation data by evaluating the children recursively
        {
            storage.values.fill(0);
            
            for(size_t c=0;c<Octree<T,DIM>::N_Children;c++)
            {
                const long int c_id=m_src_octree->child(level,id, c);		
                recursive_mult(c_id,level+1,weights,result,storage,tmp_result,transformedNodes);
            }
        }

	//before we can use the interpolation data, we habe to run a chebychev transform on it
	const size_t stride=ho_chebNodes.cols();
	Eigen::Array<T,Eigen::Dynamic,DIMOUT> tmpData(stride,DIMOUT);
	for (int memId =0;memId<storage.grid.activeCones().size();memId++) {
	    
	    ChebychevInterpolation::chebtransform<T,DIM>(storage.values.middleRows(memId*stride,stride),tmpData,high_order);
	    storage.values.middleRows(memId*stride,stride)=tmpData;
	}
	

#define USE_REINTERPOLATION	
#ifdef USE_REINTERPOLATION
	//We now reinterpolate on a finer low-order grid to get the fast point-evaluation
	ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> refinedData;
	refinedData.order = order;
	//update the interpolation grid of the result



	
        refinedData.grid = lo_grid;
        const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
        refinedData.values.resize(lo_grid.activeCones().size()*chebNodes.cols(),DIMOUT);
	for (size_t memId =0;memId<storage.grid.activeCones().size();memId++) {
	    const size_t cid=ho_grid.activeCones()[memId];
	    ConeRef hoCone({level,cid, memId,(size_t) id});
	    
	    coarseToFine(storage, level, hoCone, tmp_result, order, high_order,H,refinedData);
	}

	//before we can use the interpolation data, we habe to run a chebychev transform on it
	const size_t stride2=chebNodes.cols();
	tmpData.resize(stride2,DIMOUT);
	for (int memId =0;memId<refinedData.grid.activeCones().size();memId++) {
	    
	    ChebychevInterpolation::chebtransform<T,DIM>(refinedData.values.middleRows(memId*stride2,stride2),tmpData,order);
	    refinedData.values.middleRows(memId*stride2,stride2)=tmpData;
	}
#else
        auto& refinedData=storage;
#endif
        //Now that sthe Interpolation data storage has been prepared, we can use it to evaluate the field at the cousin nodes
	evaluateFarField(level,id, weights, result, refinedData, tmp_result);

        //propagate the interpolation data to the parent
        if( !m_src_octree->parentHasFarTargets(level,id)) //if we won't encounter any more far-fields we can stop
            return;
        
        size_t parentId = m_src_octree->parentId(level, id);
        BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
        const auto parent_center = parent_bbox.center();
        double pH = parent_bbox.sideLength();
        
        const auto p_order = parentData.order;
	const auto& p_chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(p_order);
        


        const auto& pGrid=parentData.grid;
        if(!pGrid.isEmpty()) {
	    transformedNodes.resize(DIM,p_chebNodes.cols()*pGrid.activeCones().size());
	    tmp_result.resize(transformedNodes.cols(),DIMOUT);	    
	    tmp_result.fill(0);

	    
	    size_t idx=0;

            for (int memId =0;memId<pGrid.activeCones().size();memId++) {
                const size_t el=pGrid.activeCones()[memId];               
                transformInterpToCart(pGrid.transform(el,p_chebNodes), transformedNodes.middleCols(idx,p_chebNodes.cols()),
                                      parent_center, pH);
		idx+=p_chebNodes.cols();
            }
	    
	    transferInterp(refinedData, transformedNodes, center, H, parent_center, pH,
			   tmp_result);

	    idx=0;
	    for (int memId =0;memId<pGrid.activeCones().size();memId++) {
                const size_t el=pGrid.activeCones()[memId];               
                parentData.values.middleRows(memId*stride,stride)+=tmp_result.middleRows(idx,stride);
		idx+=stride;
	    }

        }

    }


    void transferRotation(
			  const ChebychevInterpolation::InterpolationData<T, DIM, DIMOUT> data,
			  ConeRef coneRef, const ConeDomain<DIM> & domain,                        
			  const Eigen::Ref<const Eigen::Vector<double, DIM> > &direction,
                          Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result, bool backward, const Eigen::Vector<int, DIM>& target_order) const
    {

	auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double,1>(target_order.head(1));
        PointArray  targets(DIM,target_order.tail(DIM-1).prod());
        targets.topRows(1).fill(0); //data.grid.region(coneRef.id()).center()[0]
        targets.bottomRows(2)=ChebychevInterpolation::chebnodesNdd<double,2>(target_order.template tail<DIM-1>());

	PointArray  transformed(DIM,targets.cols());

	Eigen::Vector3d xc=Eigen::Vector3d::Zero();
	const double H=1;

	transformInterpToCart(domain.transform(coneRef.id(),targets), targets, xc, H);
	
	auto rotation=Eigen::Quaternion<double>::FromTwoVectors(direction,Eigen::Vector3d({0,0,1}));
	Eigen::Array<double, DIM,1> pnt;
	if(backward) {
	    rotation=rotation.inverse();
	}
	
	for(int j=0;j<targets.cols();j++) {
	    pnt=rotation*(targets.col(j));
	    targets.col(j)=pnt;
	}	       
	    
	//transform to the un-rotated interpolation domain
        transformCartToInterp(targets, transformed, xc, H);

	result.fill(0);
	const size_t stride=data.order.prod();


	const int N=targets.cols();

	std::vector<int> elIds(N);
	for(size_t idx=0;idx<N;idx++) {
	    //std::cout<<"tr"<<transformed.col(idx).transpose()<<std::endl;
	    elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	}
	std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	PointArray tmp=Util::copy_with_permutation(transformed,perm);
	size_t idx=0;
	Eigen::Array<T,Eigen::Dynamic, DIMOUT> tmp_data(result.rows(),result.cols());
	while (idx<N)
	{
	    size_t nb=1;
	    const size_t el=elIds[perm[idx]];
	    const size_t memId=data.grid.memId(el);
	    
	    //look if any of the following points are also in this element. that way we can process them together
	    while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		nb++;
	    }
	    
	    tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));

	    ChebychevInterpolation::fast_evaluate_tp<T,DIM,DIMOUT>(
    	        tmp.array().block(1,idx, 2, nb),
		chebNodes1d,
		0, //axis that is of tensor product form
		data.values.middleRows(memId*stride,stride),
		tmp_data.middleRows(idx*target_order[0],nb*target_order[0]), data.order);

	    
	    //ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(tmp.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order);
	    idx+=nb;
	}
	
	for (size_t i = 0; i < perm.size(); i++) {
	    for(size_t j=0;j<target_order[0];j++) {
		result.row(perm[i]*data.order[0]+j) = tmp_data.row(i*data.order[0]+j);
	    }
	}


#if 0
	auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double,1>(data.order.head(1));
        PointArray  targets(DIM,data.order.tail(DIM-1).prod());

        targets.topRows(1).fill(data.grid.region(coneRef.id()).center()[0]);

        targets.bottomRows(2)=ChebychevInterpolation::chebnodesNdd<double,2>(data.order.template tail<DIM-1>());
	targets=data.grid.transform(coneRef.id(),targets);

	PointArray  tmp2(DIM,targets.cols());
	transformInterpToCart(targets, tmp2, xc, H);
	//rotation that takes the vector p_xc-x_c to the Z-axis
	auto rotation=Eigen::Quaternion<double>::FromTwoVectors(direction,Eigen::Vector3d({0,0,1}));
	if(backward) {
	    rotation=rotation.inverse();
	}
	for(int j=0;j<tmp2.cols();j++) {
	    tmp2.col(j)=rotation*tmp2.col(j);
	    tmp2.col(j)*=scale;
	}

	transformCartToInterp(tmp2,targets, xc,H);
	//we now have the correct target points.			   
	//note that the first coordinate in tmp2 is to be replaced by the chebNodes as rotations do not change the radius

	result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;
	const auto& grid=data.grid;

	PointArray transformed(DIM, targets.cols());
	transformed.fill(0);

	const int N=targets.cols();
        Eigen::Vector<double,DIM> ex_point;
        size_t idx=0;
        while (idx<N)
        {
            size_t nb=0;

	    ex_point.bottomRows(2)=targets.block(1,idx,2,1);
            ex_point(0)=grid.region(coneRef.id()).center()[0]; //just use a placeholder value.

            const size_t el=data.grid.elementForPoint(ex_point);
            const size_t memId=coneRef.memId();


            //look if any of the following points are also in this element. that way we can process them together
	    for(; idx+nb<targets.cols(); nb++) {
		ex_point.bottomRows(2)=targets.block(1,idx+nb,2,1);

		if(data.grid.elementForPoint(ex_point)!=el) {
		    break;
		}
		//transformed.col(idx+nb)=data.grid.transformBackwards(el,ex_point);
	    }


	    transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,targets.middleCols(idx,nb));
            ChebychevInterpolation::fast_evaluate_tp<T,DIM,DIMOUT>(
    	        transformed.array().block(1,idx, 2, nb),
		chebNodes1d,
		0, //axis that is of tensor product form
		data.values.middleRows(memId*stride,stride),
		result.middleRows(idx*data.order[0],nb*data.order[0]), data.order);
	    	    
            idx+=nb;
        }
#endif

	
    }

    void transferTranslation(
			  const ChebychevInterpolation::InterpolationData<T, DIM, DIMOUT> data,
			  ConeRef coneRef, const ConeDomain<DIM>& grid,
                          const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                          const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                          Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
	auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double,1>(data.order.tail(1));
        PointArray  targets(DIM,data.order.head(DIM-1).prod());
        targets.row(2).fill(0);
        targets.topRows(2)=ChebychevInterpolation::chebnodesNdd<double,2>(data.order.template head<DIM-1>());
	       
	transformInterpToCart(grid.transform(coneRef.id(),targets), targets, xc-Eigen::Vector3d(0,0,(p_xc-xc).norm()), pH);
	

	PointArray transformed(DIM, targets.cols());	    
	//transform to the child interpolation domain
        transformCartToInterp(targets, transformed, xc, H);

	result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;


	const int N=targets.cols();

	std::vector<int> elIds(N);
	for(size_t idx=0;idx<N;idx++) {
	    elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	}
	std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	//std::vector<size_t> perm(elIds.size());
	//std::iota(perm.begin(),perm.end(),0);
	
	PointArray tmp=Util::copy_with_permutation(transformed,perm);
	Eigen::Array<T,Eigen::Dynamic, DIMOUT> tmp_data(result.rows(),result.cols());
	Eigen::Array<T, Eigen::Dynamic,DIMOUT> res_tmp;
  	
	size_t idx=0;
	while (idx<N)
	{
	    size_t nb=1;
	    const size_t el=elIds[perm[idx]];
	    const size_t memId=data.grid.memId(el);
	    
	    //look if any of the following points are also in this element. that way we can process them together
	    while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		nb++;
	    }
	    
	    tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));

    
	    res_tmp.resize(nb*data.order[2],DIMOUT);
	    res_tmp.fill(0);

	    
    
	    ChebychevInterpolation::fast_evaluate_tp<T,DIM,DIMOUT>(
    	        tmp.array().block(0,idx, 2, nb),
		chebNodes1d,
		2, //axis that is of tensor product form
		data.values.middleRows(memId*stride,stride),
		res_tmp, data.order);



	    for(size_t l=0;l<chebNodes1d.cols();l++) {
		for (size_t i = 0; i < nb; i++) {
		    result.row(l*N+perm[idx+i])=res_tmp.row(l*nb +i);
		}
	    }

	    idx+=nb;
	}

	
#if 0
	auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double, 1>(data.order.tail(1));
        Eigen::Array<double,DIM, -1>  targets(DIM,data.order.head(DIM-1).prod());
        targets.bottomRows(1).fill(data.grid.region(coneRef.id()).center()[2]);
        targets.topRows(2)=ChebychevInterpolation::chebnodesNdd<double,2>(data.order.template head<DIM-1>());

	BoundingBox<DIM> target_domain=data.grid.domain();
	//target_domain.max()[0]=sqrt(DIM)/DIM;

	Eigen::Array<double,DIM, -1>  tmp2(DIM,targets.cols());
	
	ConeDomain target_grid=data.grid;//(data.grid.n_elements(),target_domain);
	Eigen::Vector3d xc2=xc;
	xc2[2]-=(p_xc-xc).norm();
	targets=target_grid.transform(coneRef.id(),targets);
	transformInterpToCart(targets, tmp2, xc2, pH);

	//transform the points back to the original points
	transformCartToInterp(tmp2,targets, xc,H);
	
	//we now have the correct target points in the original interp-coordinates.			   
	//note that the first coordinate in tmp2 is to be replaced by the chebNodes as rotations do not change the radius

	PointArray transformed(DIM, targets.cols());

	result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;
	const auto& grid=data.grid;

	const int N=targets.cols();
        Eigen::Vector<double,DIM> ex_point;
        size_t idx=0;
	Eigen::Array<T, Eigen::Dynamic,DIMOUT> res_tmp;
	result.fill(0);
        while (idx<N)
        {
            size_t nb=0;
	    ex_point.topRows(2)=targets.block(0,idx,2,1);
            ex_point(2)=grid.region(coneRef.id()).center()[2]; //just use a placeholder value.

	    //std::cout<<"ex"<<ex_point.transpose()<<" "<<grid.domain()<<std::endl;
            const size_t el=data.grid.elementForPoint(ex_point);
	    //std::cout<<"done"<<std::endl;
            const size_t memId=coneRef.memId();

            
            //look if any of the following points are also in this element. that way we can process them together
	    for(; idx+nb<targets.cols(); nb++) {		
		ex_point.topRows(2)=targets.block(0,idx+nb,2,1);
		//std::cout<<"ex"<<ex_point.transpose()<<" "<<grid.domain()<<std::endl;
		if(data.grid.elementForPoint(ex_point)!=el) {
		    break;
		}
		//transformed.col(idx+nb)=data.grid.transformBackwards(el,ex_point);        

	    }


	    res_tmp.resize(nb*data.order[2],DIMOUT);
	    res_tmp.fill(0);
	    /*PointArray tts(chebNodes1d.cols(),DIM);
	    for(int i=0;i<chebNodes1d.cols();i++) {
		tts.col(i)=ex_point;
		tts(i,2)=chebNodes1d(i);
	    }
	    
	    
	    ChebychevInterpolation::parallel_evaluate(
						      tts,
						      data.values.middleRows(memId*stride,stride),
						      res_tmp, data.order, data.grid.region(el)); */

	    transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,targets.middleCols(idx,nb));

            ChebychevInterpolation::fast_evaluate_tp<T,DIM,DIMOUT>(
                transformed.array().block(0,idx, 2, nb),
		chebNodes1d,
		2, //axis that is of tensor product form
		data.values.middleRows(memId*stride,stride),
		res_tmp, data.order);

	    for(size_t l=0;l<chebNodes1d.cols();l++) {
		result.middleRows(l*N+idx,nb)=res_tmp.middleRows(l*nb,nb);			
	    }
            idx+=nb;
        }
#endif
    
	
    }
    
    void coarseToFine(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& interpolationData, size_t level, const ConeRef& hoCone, 
		      Eigen::Array<T, Eigen::Dynamic,1>& tmp_result, const Eigen::Vector<int, DIM>& order, const Eigen::Vector<int, DIM>& high_order,double H0,
		      ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& result)
    {
#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem coarse to fine");
	ngcore::RegionTimer reg(t);
#endif

	size_t boxId=hoCone.boxId();
	const auto& hoGrid= m_src_octree->coneDomain(level,boxId,1);
	
	if (!m_src_octree->hasPoints(level, boxId)) {
	    return;
	}

	//BoundingBox region=//hoGrid.region(hoCone.id());
	if(! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
	    return;
	}


	auto fine_N=static_cast<Derived *>(this)->elementsForBox(H0, this->m_baseOrder,this->m_base_n_elements,0);
	Eigen::Vector<size_t,DIM> factor=fine_N.array()/hoGrid.num_elements().array();

	
	std::array<Eigen::Array<double,Eigen::Dynamic,1>, DIM > points;
	size_t Np=1;
	for(int d=0;d<DIM;d++) {
	    auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double,1>(Eigen::Vector<int,1>(order[d]));
	    points[d].resize(chebNodes1d.size()*factor[d]);

	    Np*=points[d].size();

	    for(int j=0;j<factor[d];j++){
		const auto h=2;
		auto min=-1+(j*(h/((double) factor[d])));
		auto max=(min+(h/((double) factor[d])));
		const double a=0.5*(max-min);
		const double b=0.5*(max+min);


		points[d].segment(j*chebNodes1d.size(),chebNodes1d.size())=(chebNodes1d.array()*a)+b;
	    }
	}

	tmp_result.resize(Np,DIMOUT);
	const size_t ho_stride=high_order.prod();
	ChebychevInterpolation::tp_evaluate<T,DIM,DIMOUT>(points, interpolationData.values.middleRows(hoCone.memId()*ho_stride,ho_stride),tmp_result, high_order);
		    
	
	
	Eigen::Vector<double,DIM> pnt;
	size_t fine_stride=order.prod();
	size_t idx_coarse=0;
	//now distribute the results to the right places. Do it in a slow but safe way		    
	for(int l=0;l<factor[2]*order[2];l++) {
	    for(int j=0;j<factor[1]*order[1];j++) {
		for(int k=0;k<factor[0];k++) //the innermost dimension is continuous in memory so we do things blocked
		{
				
		    auto ho_id=hoGrid.indicesFromId(hoCone.id());

		    const size_t fine_el=
			(ho_id[2]*factor[2]+(l/order[2]))*result.grid.n_elements(1)*result.grid.n_elements(0)+
			(ho_id[1]*factor[1]+(j/order[1]))*result.grid.n_elements(0)+
			(ho_id[0]*factor[0]+(k));
				
		    //auto fine_el2=result[boxId].grid.elementForPoint(pnt);

		    //std::cout<<"fine_el"<<fine_el<<" "<<fine_el2<<" "<<hoCone.id()<<" "<<ho_id.transpose()<<" ljk:"<<l<<" "<<j<<" "<<k<<std::endl;
		    //assert(fine_el==fine_el2);
				
		    if(result.grid.isActive(fine_el)) { //if the cone is not active, discard it
			size_t fine_memId=result.grid.memId(fine_el);
			size_t idx_fine=(l % order[2])*order[0]*order[1]+(j % order[1])*order[0];// + (k % order[0]);

			result.values.middleRows(fine_memId*fine_stride+idx_fine, order[0])=tmp_result.middleRows(idx_coarse,order[0]);
		    }
		    idx_coarse+=order[0];
				
			    
		}

	    }
	}

    }

  
    void transferInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data, const Eigen::Ref<const PointArray> &targets,
                        const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                        const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                        Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem transfer interp");
	ngcore::RegionTimer reg(t);
#endif

        //transform to the child interpolation domain
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

	//result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;
	const auto& grid=data.grid;

	const int N=targets.cols();
	if(false) {
	    size_t idx=0;
	    while (idx<N)
	    {
		size_t nb=1;
		//std::cout<<"\t box: "<<data.grid.domain()<<std::endl;

		const size_t el=data.grid.elementForPoint(transformed.col(idx));
		const size_t memId=data.grid.memId(el);

		//transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
		//look if any of the following points are also in this element. that way we can process them together
		while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
		    //transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));		
		    nb++;
		}
		//transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,transformed.middleCols(idx,nb));
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(transformed.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order,
									 data.grid.region(el));
		idx+=nb;
	    }
	}else{
	    std::vector<int> elIds(N);
	    for(size_t idx=0;idx<N;idx++) {
		elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	    }
	    std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	    PointArray tmp=Util::copy_with_permutation(transformed,perm);
	    size_t idx=0;
	    while (idx<N)
	    {
		size_t nb=1;
		const size_t el=elIds[perm[idx]];
		const size_t memId=data.grid.memId(el);

		//look if any of the following points are also in this element. that way we can process them together
		while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		    nb++;
		}

		tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(tmp.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order);
		idx+=nb;
	    }
	    result=Util::copy_with_inverse_permutation(result,perm);
	}
	
        
	static_cast<const Derived *>(this)->transfer_factor(targets, xc, H, p_xc, pH, result);	
    }


    //evaluateFromIntepr simplified codepath if we now we are dealing with a single target
    void inline evaluateSingleFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data,
				  const Eigen::Ref<const Eigen::Array<double,DIM,1> > &target,
				  const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
				  Eigen::Ref<Eigen::Array<T, 1, DIMOUT> > result) const
    {

	Eigen::Array<double,DIM, 1> transformed(DIM);
	transformCartToInterp(target, transformed, xc, H);
	const size_t stride=data.computeStride();
	const size_t el=data.grid.elementForPoint(transformed);
	const size_t memId=data.grid.memId(el);

	transformed=data.grid.transformBackwards(el,transformed);
	ChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT, -1,-1,-1> clenshaw;
	result=clenshaw(transformed, data.values.middleRows(memId*stride,stride),  data.order);
	
	
	const auto cf = static_cast<const Derived *>(this)->CF(target.matrix() - xc);
	result*= cf;

    }


    
    void evaluateFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data,
                            const Eigen::Ref<const PointArray> &targets,
                            const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                            Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
	if(targets.cols()==0) {
	    return;
	}
	
	//std::cout<<"efromInt"<<data.order<<std::endl;
	//sort the points into the corresponding cones
	const int N=targets.cols();
	PointArray transformed(DIM, targets.cols());
	transformCartToInterp(targets, transformed, xc, H);
	const size_t stride=data.computeStride();
	//std::cout<<"stride"<<stride<<std::endl;
	/*	assert(data.order==7);
		const static Eigen::Vector<double,7> nodes = ChebychevInterpolation::chebnodes1d<double, 7>();
	
	for(size_t idx=0;idx<transformed.cols();idx++) {
	    const size_t el=data.grid.elementForPoint(transformed.col(idx));	    
	    const size_t memId=data.grid.memId(el);

	    transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	    result(idx)=ChebychevInterpolation::evaluate_slow<T, 1, 7, DIM>(transformed.array().col(idx), data.values.segment(memId*stride,stride),nodes)[0];
	    const auto cf = static_cast<const Derived *>(this)->CF(targets.col(idx) - xc);
            result[idx] *= cf;
	}*/


	if(true) {
	    size_t idx=0;
	    size_t nextElement=data.grid.elementForPoint(transformed.col(0));
	    while (idx<N)
	    {
		size_t nb=0;
		const size_t el=nextElement;
		const size_t memId=data.grid.memId(el);

		//transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
		//look if any of the following points are also in this element. that way we can process them together
		while(nextElement==el) {
		    nb++;
		    if(idx+nb<transformed.cols())  {
			nextElement=data.grid.elementForPoint(transformed.col(idx+nb));
		    }else{
			break;
		    }
		    //transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));		
		}
		transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,transformed.middleCols(idx,nb));

		

		//result.row(idx)=ChebychevInterpolation::evaluate_clenshaw<T, 1,DIM,DIMOUT>(transformed.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride),  data.order);
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(transformed.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order);
		idx+=nb;
	    }
	    for (unsigned int j = 0; j < targets.cols(); j++) {
		const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j).matrix() - xc);
		result.row(j) *= cf;
	    }

	}else{
	    std::vector<int> elIds(N);
	    for(size_t idx=0;idx<N;idx++) {
		elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	    }
	    std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	    PointArray tmp=Util::copy_with_permutation(transformed,perm);
	    size_t idx=0;
	    while (idx<N)
	    {
		size_t nb=1;
		const size_t el=elIds[perm[idx]];
		const size_t memId=data.grid.memId(el);

		//look if any of the following points are also in this element. that way we can process them together
		while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		    nb++;
		}
		//tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(tmp.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order,
									 data.grid.region(el));
		idx+=nb;
	    }

	    result=Util::copy_with_inverse_permutation(result,perm);
	    for (unsigned int j = 0; j < targets.cols(); j++) {
		const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j).matrix() - xc);
		result.row(j) *= cf;
	    }

	}

	
	// size_t idx=0;
	// while (idx<transformed.cols())
	// {
	//     //std::cout<<"idx"<<idx<<std::endl;
	//     size_t nb=1;	    
	//     const size_t el=data.grid.elementForPoint(transformed.col(idx));	    
	//     const size_t memId=data.grid.memId(el);
	   

	//     //std::cout<<"el="<<el<<" "<<transformed.col(idx)<<std::endl;
	//     transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	//     //look if any of the following points are also in this elemnt. that way we can process them together
	//     while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
	// 	transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));
	// 	nb++;
	//     }
	//     ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array().middleCols(idx,nb), data.values.segment(memId*stride,stride), result.segment(idx,nb), data.order);
	//     idx+=nb;
	// }
	
		
	//std::cout<<"done"<<data.order<<std::endl;
    }


    inline void transformCartToInterp(const Eigen::Ref<const PointArray > &nodes,
				      Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {
	Util::cartToInterp2<DIM>(nodes.array(), xc, H,transformed);
        /*for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = Util::cartToInterp<DIM>(nodes.col(i), xc, H);
	    }*/
    }

    inline void transformInterpToCart(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    { 

	transformed = Util::interpToCart<DIM>(nodes.array(), xc, H);
       /*for (int i = 0; i < nodes.cols(); i++) {
	 transformed.col(i) = Util::interpToCart<DIM>(nodes.col(i), xc, H);
	 }*/
    }

    

protected:
    void onOctreeReady()
    {
	//do nothing. but give subclasses the opportunity to initialize some things
    }

private:
    std::unique_ptr<Octree<T, DIM> > m_src_octree;
    std::unique_ptr<Octree<T, DIM> > m_target_octree;
    unsigned int m_numTargets;
    unsigned int m_numSrcs;
    Eigen::Vector<size_t, DIM> m_base_n_elements;
    size_t m_baseOrder;
    double m_tolerance;
#ifdef  CHECK_CONNECTIVITY
    tbb::queuing_mutex m_conMutex;
    Eigen::MatrixXi m_connectivity;
#endif
};

#endif
