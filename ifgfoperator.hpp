#ifndef __IFGFOPERATOR_HPP_
#define __IFGFOPERATOR_HPP_

#include <Eigen/Dense>
#include "octree.hpp"
#include "chebinterp.hpp"
#include <tbb/queuing_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>

#include <fstream>
#include <iostream>


//#define CHECK_CONNECTIVITY
//#define TWO_GRID_ONLY
#define  RECURSIVE_MULT

#include <memory>

template<typename T, unsigned int DIM, unsigned int DIMOUT, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Array<double, DIM, Eigen::Dynamic> PointArray;     //, Eigen::RowMajor?

    enum RefinementType { RefineH, RefineP};

    IfgfOperator(long int maxLeafSize = -1, size_t order=5, size_t n_elements=1, double tolerance = -1)
    {
	if constexpr (DIM==3) {
	    m_base_n_elements[0]=1;
	    m_base_n_elements[1]=2;
	    m_base_n_elements[2]=4;
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
	m_src_octree->calculateInterpolationRange([this](double H){return static_cast<Derived *>(this)->orderForBox(H, this->m_baseOrder);},
						  [this](double H){return static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements);},*m_target_octree);

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
	const int maxR=refine== RefineH ? 4 : 15;
	
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

	PointArray samplePoints=PointArray::Random(DIM,n_samples);		
	PointArray transformedSample(DIM,samplePoints.cols());

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

        tbb::queuing_mutex resultMutex;
	
        tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > tmp_result;
	tbb::enumerable_thread_specific<PointArray > transformedNodes;

#ifdef CHECK_CONNECTIVITY
	std::cout<<"srcs="<<m_numSrcs<<" "<<m_numTargets<<std::endl;
	m_connectivity.resize(m_numSrcs, m_numTargets);
        m_connectivity.fill(0);
	std::cout<<"done"<<std::endl;
#endif
#ifndef RECURSIVE_MULT
	std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > interpolationData(m_src_octree->numBoxes(level));
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > parentInterpolationData;
#else
        //depending on how parallell we are, we use the recursive or the iterative way of
        //iterating over all the boxes (depth first vs breath first). This saves some money
        int min_boxes=16;
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
	    if(m_src_octree->farTargets(level-1,pId).size()==0 && !m_src_octree->parentHasFarTargets(level-1,pId)) {
		continue;
	    }
	    BoundingBox bbox = m_src_octree->bbox(level - 1, pId);
	    double H = bbox.sideLength();
	    order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder);
	    auto grid= m_src_octree->coneDomain(level-1,pId);		
	    interpolationData[pId].grid = grid;
	    interpolationData[pId].values.resize(grid.activeCones().size()*order.prod(),DIMOUT);            
	    interpolationData[pId].values.fill(0);
	    interpolationData[pId].order = order;                        
        }

        
	tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numBoxes(level)),
        [&](tbb::blocked_range<size_t> r) {
            for(size_t id=r.begin();id<r.end();id++) {
                recursive_mult(id, level, new_weights,result, interpolationData[m_src_octree->parentId(level, id)],
			       tmp_result.local(), resultMutex, transformedNodes.local());                
            }});
        
        /*for(size_t id=0; id < m_src_octree->numBoxes(level); id++)
          recursive_mult(id, level, new_weights,result, interpolationData[m_src_octree->parentId(level, id)],
	  tmp_result.local(), resultMutex, transformedNodes.local());                */
        
	level--;
        std::cout<<"now proceeding iteratively"<<level<<std::endl;
#endif

        for (; level >= 0; --level) {
            std::cout << "level=" << level << " "<< m_src_octree->numBoxes(level)<< std::endl;
	    

            std::cout << "step 1" <<std::endl;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numBoxes(level)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		if(!m_src_octree->hasPoints(level,i))
		    continue;
		BoundingBox bbox = m_src_octree->bbox(level, i);
		auto center = bbox.center();
		double H = bbox.sideLength();
		const auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder);

		const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
		transformedNodes.local().resize(DIM,chebNodes.cols());;
		
		evaluateNearField(level,i, new_weights, result, tmp_result.local(), resultMutex,transformedNodes.local());
#ifndef RECURSIVE_MULT
		if(m_src_octree->isLeaf(level,i)) {
		    auto grid=m_src_octree->coneDomain(level,i);

		    //std::cout<<"grid="<<grid.domain().min().transpose()<<std::endl;
		    interpolationData[i].order = order;
		    interpolationData[i].grid=grid;

		    

		    IndexRange srcs=m_src_octree->points(level,i);
		    const size_t nS=srcs.second-srcs.first;

		    //std::cout<<"interp"<<std::endl;
		    interpolationData[i].values.resize(grid.activeCones().size()*chebNodes.cols(),DIMOUT);

		    const size_t stride=chebNodes.cols();
		    for (int memId =0;memId<interpolationData[i].grid.activeCones().size();memId++) {
			//std::cout<<"mem"<<memId;
			const size_t el=interpolationData[i].grid.activeCones()[memId];

			transformInterpToCart(grid.transform(el,chebNodes), transformedNodes.local(), center, H);
			interpolationData[i].values.middleRows(memId*stride,stride) =
			    static_cast<const Derived *>(this)
			    ->evaluateFactoredKernel(m_src_octree->points(srcs),
						     transformedNodes.local(),
						     new_weights.segment(srcs.first, nS), center, H,srcs);

			
		    }
		}
#endif
		//before we can use the interpolation data, we habe to run a chebychev transform on it
		const size_t stride=chebNodes.cols();
		Eigen::Array<T,Eigen::Dynamic,DIMOUT> tmpData(stride,DIMOUT);
		for (int memId =0;memId<interpolationData[i].grid.activeCones().size();memId++) {

		    ChebychevInterpolation::chebtransform<T,DIM>(interpolationData[i].values.middleRows(memId*stride,stride),tmpData,order);
		    interpolationData[i].values.middleRows(memId*stride,stride)=tmpData;
		}
		evaluateFarField(level,i, new_weights, result, interpolationData[i], tmp_result.local(), resultMutex);


            }});
            std::cout << "step 2" << std::endl;

            //std::cout<<"connectivity"<<std::endl<<m_connectivity<<std::endl;

	    if(level==0) {
		continue;
	    }
            //Now transform the interpolation data to the parents	    	
            parentInterpolationData.resize(m_src_octree->numBoxes(level - 1));
	    std::vector<tbb::queuing_mutex> interpDataMutex(m_src_octree->numBoxes(level - 1));
            for (int pId = 0; pId < parentInterpolationData.size(); pId++) {		
		if (!m_src_octree->hasPoints(level-1, pId)) {
                        continue;
		}

                BoundingBox bbox = m_src_octree->bbox(level - 1, pId);
		//std::cout<<"bbox="<<bbox.min().transpose()<<" "<<bbox.max().transpose()<<std::endl;
                auto center = bbox.center();
                double H = bbox.sideLength();

                const auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder);
		auto grid= m_src_octree->coneDomain(level-1,pId);		
		parentInterpolationData[pId].grid = grid;
		parentInterpolationData[pId].values.resize(grid.activeCones().size()*order.prod(),DIMOUT);
                //parentInterpolationData[pId].values.resize(ChebychevInterpolation::chebnodesNdd<double, DIM>(order).cols());
                parentInterpolationData[pId].values.fill(0);
                parentInterpolationData[pId].order = order;

		//std::cout<<"Interpolation"<<order<<std::endl;

            }
	    

            std::cout << "step 3" << std::endl;
	    
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numBoxes(level),32),
            [&](tbb::blocked_range<size_t> r) {
                PointArray chebNodes;// = ChebychevInterpolation::chebnodesNdd<double, DIM>(m_baseOrder);
                Eigen::Matrix<T, Eigen::Dynamic,DIMOUT> tmpInterpolationData;
		Eigen::Vector<int, DIM> old_p_order;
		old_p_order.fill(-1);
		int skipped=0;
                for (size_t i = r.begin(); i < r.end(); i++)
                {
                    if (!m_src_octree->hasPoints(level, i)) {
			//skipped++;
                        continue;
                    }

		    if(level<2 && ! m_src_octree->parentHasFarTargets(level, i)){ //we dont need the interpolation info for those levels.
			continue;
		    }
		    
                    //current node
                    BoundingBox bbox = m_src_octree->bbox(level, i);
                    auto center = bbox.center();
                    double H = bbox.sideLength();

                    //parent node
                    size_t parentId = m_src_octree->parentId(level, i);
                    BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
                    auto parent_center = parent_bbox.center();
                    double pH = parent_bbox.sideLength();

                    const auto p_order = parentInterpolationData[parentId].order;
                    if (p_order != old_p_order) {
                        chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(p_order);
                        transformedNodes.local().resize(DIM, chebNodes.cols());

                        old_p_order = p_order;
                    }


		    auto grid=parentInterpolationData[parentId].grid;
                    tmpInterpolationData.resize(grid.activeCones().size()*chebNodes.cols(),DIMOUT);
                    tmpInterpolationData.fill(0);
#ifdef TWO_GRID_ONLY                    	    
		    const size_t stride=chebNodes.cols();

		    if(!grid.isEmpty()) {
                        IndexRange srcs = m_src_octree->points(level, i);
                        int nS = srcs.second - srcs.first;

			for (int memId =0;memId<grid.activeCones().size();memId++) {
			    const size_t el=grid.activeCones()[memId];
			    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes.local(), parent_center, pH);
			    tmpInterpolationData.middleRows(memId*stride,stride) =
				static_cast<const Derived *>(this)->evaluateFactoredKernel(m_src_octree->points(srcs), transformedNodes, new_weights.segment(srcs.first, nS), parent_center, pH,srcs);
			}
		    }

                    //tmpInterpolationData =
                    //    static_cast<const Derived *>(this)->evaluateFactoredKernel(m_src_octree->points(srcs), transformedNodes, new_weights.segment(srcs.first, nS), parent_center, pH);
#else

		    if(!grid.isEmpty()) {
			const size_t stride=chebNodes.cols();
			for (int memId =0;memId<grid.activeCones().size();memId++) {
			    const size_t el=grid.activeCones()[memId];
			    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes.local(), parent_center, pH);

			    transferInterp(interpolationData[i], transformedNodes.local(), center, H, parent_center, pH, tmpInterpolationData.middleRows(memId*stride,stride));
			    
			}
			//char a;
			//std::cin>>a;
		    }

#endif

                    //Free the data we no longer use
                    interpolationData[i].values.resize(0,DIMOUT);
                    {
                        assert(parentInterpolationData[parentId].values.size() == tmpInterpolationData.size());
                        tbb::queuing_mutex::scoped_lock lock(interpDataMutex[parentId]);
                        parentInterpolationData[parentId].values.matrix() += tmpInterpolationData;
                    }
                
		//std::cout<<"skipped "<<skipped<<" boxes out of"<<r.end()-r.begin()<<std::endl;
		}});

            std::swap(interpolationData, parentInterpolationData);
            parentInterpolationData.resize(0);

        }
#ifdef CHECK_CONNECTIVITY
	if((m_connectivity.array() - 1).matrix().norm() > 1e-10) {
	    std::cout<<m_connectivity<<std::endl;
	}
        assert((m_connectivity.array() - 1).matrix().norm() < 1e-10);
#endif
        std::cout<<"done"<<std::endl;
        return Util::copy_with_inverse_permutation(result, m_target_octree->permutation());
    }


    void evaluateNearField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			  Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,
			  tbb::queuing_mutex& resultMutex,
			  PointArray &transformedNodes
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
	transformedNodes.resize(DIM, chebNodes.cols());
	
	std::vector<IndexRange> targetList = m_src_octree->nearTargets(level, id);

	for (const auto &targets : targetList) {
	    const size_t nT = targets.second - targets.first;
#ifdef CHECK_CONNECTIVITY
	    for (int l = targets.first; l < targets.second; l++) {
		for (int k = srcs.first; k < srcs.second; k++) {
		    /*if(m_connectivity(k,l)!=0) {
			std::cout<<"near error"<<k<<" "<<l<<std::endl;
			}*/
		    m_connectivity(k, l) += 1;
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
		tbb::queuing_mutex::scoped_lock lock(resultMutex);
		result.middleRows(targets.first, nT) += tmp_result;
	    }
	    
	}	

    }

    void evaluateFarField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			  ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& interpolationData,
			  Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,
			  tbb::queuing_mutex& resultMutex
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
	    for (int q = cousinTargets[l].first; q < cousinTargets[l].second; q++) {
		for (int k = srcs.first; k < srcs.second; k++) {
		    /*if(m_connectivity(q,k)!=0) {
			std::cout<<"farerror"<<q<<" "<<k<<std::endl;
			}*/

		    m_connectivity(k, q) += 1;
		}
	    }
#endif
	    tmp_result.resize(nT,DIMOUT);
			
	    evaluateFromInterp(interpolationData, m_target_octree->points(cousinTargets[l]), center, H,
			       tmp_result);
	    
	    {
		tbb::queuing_mutex::scoped_lock lock(resultMutex);
		result.middleRows(cousinTargets[l].first, nT) += tmp_result;
	    }
	}  
    }



    
    inline void recursive_mult(long int id ,size_t level,
                               const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
                               Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
                               ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& parentData,
                               Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,
                               tbb::queuing_mutex& resultMutex,
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

        
        auto grid=m_src_octree->coneDomain(level,id);

        //std::cout<<"grid="<<grid.domain().min().transpose()<<std::endl;
        storage.order = order;
        storage.grid = grid;
        const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
        storage.values.resize(grid.activeCones().size()*chebNodes.cols(),DIMOUT);


	//evaluate the near field exactly
	evaluateNearField(level,id,weights, result, tmp_result, resultMutex,transformedNodes);	
                
        //if we are at the leaf-level compute the interpolation data by actually interpolating the true function
        if(m_src_octree->isLeaf(level,id)) {                                
            //std::cout<<"interp"<<level<<std::endl;
            const size_t stride=chebNodes.cols();
            for (int memId =0;memId<grid.activeCones().size();memId++) {
                //std::cout<<"mem"<<memId;
                const size_t el=grid.activeCones()[memId];
		
                transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, center, H);
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
                recursive_mult(c_id,level+1,weights,result,storage,tmp_result,resultMutex,transformedNodes);
            }
        }

	//before we can use the interpolation data, we habe to run a chebychev transform on it
	const size_t stride=chebNodes.cols();
	Eigen::Array<T,Eigen::Dynamic,DIMOUT> tmpData(stride,DIMOUT);
	for (int memId =0;memId<storage.grid.activeCones().size();memId++) {
	    
	    ChebychevInterpolation::chebtransform<T,DIM>(storage.values.middleRows(memId*stride,stride),tmpData,order);
	    storage.values.middleRows(memId*stride,stride)=tmpData;
	}
	


        //Now that sthe Interpolation data storage has been prepared, we can use it to evaluate the field at the cousin nodes
	evaluateFarField(level,id, weights, result, storage, tmp_result, resultMutex);

        //propagate the interpolation data to the parent
        if(level <=2 && !m_src_octree->parentHasFarTargets(level,id)) //if we won't encounter any more far-fields we can stop
            return;
        
        size_t parentId = m_src_octree->parentId(level, id);
        BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
        const auto parent_center = parent_bbox.center();
        double pH = parent_bbox.sideLength();
        
        const auto p_order = parentData.order;
	const auto& p_chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(p_order);
        


        const auto& pGrid=parentData.grid;

#ifdef TWO_GRID_ONLY              
        if(!pGrid.isEmpty()) {
            for (int memId =0;memId<pGrid.activeCones().size();memId++) {
                const size_t el=pGrid.activeCones()[memId];
                transformInterpToCart(pGrid.transform(el,p_chebNodes), transformedNodes, parent_center, pH);
                parentData.values.middleRows(memId*stride,stride).matrix() +=
                    static_cast<const Derived *>(this)->evaluateFactoredKernel(m_src_octree->points(srcs),
                                                                               transformedNodes,
                                                                               weights.segment(srcs.first, nS),
                                                                               parent_center, pH,srcs);
            }
        }
        
#else
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
	    
	    transferInterp(storage, transformedNodes, center, H, parent_center, pH,
			   tmp_result);

	    idx=0;
	    for (int memId =0;memId<pGrid.activeCones().size();memId++) {
                const size_t el=pGrid.activeCones()[memId];               
                parentData.values.middleRows(memId*stride,stride)+=tmp_result.middleRows(idx,stride);
		idx+=stride;
	    }
        }
#endif
    }

    void transferInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data, const Eigen::Ref<const PointArray> &targets,
			       const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                               const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                               Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
        //transform to the child interpolation domain
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

	result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;
	const auto& grid=data.grid;

	const int N=targets.cols();
	if(true) {
	    size_t idx=0;
	    while (idx<N)
	    {
		size_t nb=1;
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
    
    void evaluateFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data,
                            const Eigen::Ref<const PointArray> &targets,
                            const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                            Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {

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
	    while (idx<N)
	    {
		size_t nb=1;
		const size_t el=data.grid.elementForPoint(transformed.col(idx));
		const size_t memId=data.grid.memId(el);

		//transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
		//look if any of the following points are also in this element. that way we can process them together
		while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
		    //transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));		
		    nb++;
		}
		transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,transformed.middleCols(idx,nb));
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
    Eigen::MatrixXi m_connectivity;
#endif
};

#endif
