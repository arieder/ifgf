#ifndef _OCTREE_HPP_
#define _OCTREE_HPP_

#include <Eigen/Dense>
#include <memory>
#include <map>
#include <vector>
#include <execution>
#include <iostream>
#include <unordered_set>

#include "util.hpp"
#include "boundingbox.hpp"
#include "zorder_less.hpp"
#include "chebinterp.hpp"
#include "cone_domain.hpp"

#include <tbb/queuing_mutex.h>
#include <tbb/spin_mutex.h>
typedef std::pair<size_t, size_t> IndexRange;

//#define  EXACT_INTERP_RANGE


template<typename T, size_t DIM>
class Octree
{

public:
    enum {N_Children = 1 << DIM };
    enum TransformationMode { Decomposition, TwoGrid , Regular};
    typedef Eigen::Array<double, DIM, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double, DIM> Point;

    class OctreeNode
    {
    private:
        std::weak_ptr<OctreeNode > m_parent;
        long int m_id;
        std::shared_ptr<OctreeNode> m_children[N_Children];
        //std::vector<std::shared_ptr<const OctreeNode> > m_neighbors;

	std::vector<IndexRange> m_farTargets;
	std::vector<IndexRange> m_nearTargets;

        IndexRange m_pntRange;

	std::array<ConeDomain<DIM>,2> m_coneDomain;

        BoundingBox<DIM> m_bbox;


        bool m_isLeaf;
        unsigned int m_level;
    public:

        OctreeNode(std::shared_ptr<OctreeNode> parent, unsigned int level) :
            m_parent(parent),
            m_isLeaf(true),
            m_level(level)
        {

        }

        ~OctreeNode()
        {	    
        }

        void setId(size_t id)
        {
            m_id = id;
        }

        long int id() const
        {
            return m_id;
        }

        IndexRange pntRange() const
        {
            return m_pntRange;
        }


        void setPntRange(const IndexRange &range)
        {
            m_pntRange = range;

        }

        void setChild(size_t idx, std::shared_ptr<OctreeNode> node)
        {
	    if(node!=0) {
		m_children[idx] = node;
		m_isLeaf = false;
	    }
        }

	void setConeDomain(const ConeDomain<DIM>& domain, int substep=0)
	{
	    m_coneDomain[substep]=domain;
	}

	const ConeDomain<DIM>& coneDomain(int substep=0) const
	{
	    return m_coneDomain[substep];
	}


	BoundingBox<DIM> interpolationRange(int substep=0) const
	{
	    return m_coneDomain[substep].domain();
	}

        const std::shared_ptr<const OctreeNode> child(size_t idx) const
        {
            return m_children[idx];
        }

        std::shared_ptr<OctreeNode> child(size_t idx)
        {
            return m_children[idx];
        }

        BoundingBox<DIM> boundingBox() const
        {
            return m_bbox;
        }

        void setBoundingBox(const BoundingBox<DIM> &bbox)
        {
            m_bbox = bbox;
        }

        bool isLeaf() const
        {
            return m_isLeaf;
        }

        bool hasPoints() const
        {            
            return m_pntRange.first != m_pntRange.second;
        }

        unsigned int level() const
        {
            return m_level;
        }


        const std::weak_ptr<const OctreeNode> parent() const
        {
            return m_parent;
        }

	void addNearInteraction(const OctreeNode& target)
	{
	    if(target.hasPoints())
		m_nearTargets.push_back(target.pntRange());
	}
	
	void addFarInteraction(const OctreeNode& target)
	{
	    if(target.hasPoints())
		m_farTargets.push_back(target.pntRange());
	}

	

	const std::vector<IndexRange>& nearTargets() const
	{
	    return m_nearTargets;
	}

	const std::vector<IndexRange>& farTargets() const
	{
	    return m_farTargets;
	}

	
	
        void print(std::string prefix = "") const
        {
            std::cout << prefix;
            std::cout << "----";

            std::cout << m_id << " ";
            std::cout << "(" << m_pntRange.first << " " << m_pntRange.second << ")"; //<<std::endl;
            //std::cout << m_bbox.min().transpose() << "   " << m_bbox.max().transpose() << std::endl;
	    std::cout<< m_isLeaf <<std::endl;

            //std::cout<<"("<<m_pntRange.first<<" "<<m_pntRange.second<<")"<<std::endl;
            //std::cout<<" "<<m_bbox.min().transpose()<<" to "<<m_bbox.max().transpose()<<std::endl;
            /*std::cout<<m_id<<" ";
            if(m_parent)
            std::cout<<m_parent->id();
            std::cout<<std::endl;*/
            if (!m_isLeaf) {
                for (int i = 0; i < N_Children; i++) {
                    if (m_children[i] != 0) {
                        m_children[i]->print(prefix + "    ");
                    } else {
                        std::cout << prefix + "    ----x" << std::endl;
                    }
                }
            }
        }

    };

    typedef std::pair<size_t, size_t> BoxIndex; //level + index in level

    Octree(int maxLeafSize):
        m_maxLeafSize(maxLeafSize)
    {

	double eta=(double) sqrt((double) DIM);
	m_isAdmissible= [eta] (const BoundingBox<DIM>& src,const BoundingBox<DIM>& target) { return target.exteriorDistance(src.center()) >= eta* src.sideLength();};

    }

    ~Octree()
    {

    }

    void build(const PointArray &pnts)
    {
        std::cout << "building a new octree" << pnts.cols()<<std::endl;

        std::cout << "finding bbox" << std::endl;
        Point min = pnts.col(0);
        Point max = pnts.col(0);

        for (size_t i = 0; i < pnts.cols(); i++) {
            min = min.array().min(pnts.col(i).array());
            max = max.array().max(pnts.col(i).array());
        }


        //make the bbox slightly larger to not have to deal with boundary issues
        min.array() -= 1e-8 * min.norm();
        max.array() += 1e-8 * max.norm();

        BoundingBox<DIM> bbox(min, max);
        std::cout << "bbox=" << min.transpose() << "\t" << max.transpose() << std::endl;

        //sort the points by their morton order for better locality later on
        std::cout << "sorting..." << std::endl;
        m_permutation = Util::sort_with_permutation( pnts.colwise().begin(), pnts.colwise().end(), zorder_knn::Less<Point, DIM>(bbox));
        m_pnts = Util::copy_with_permutation(pnts, m_permutation);

	m_diameter=bbox.diagonal().norm();

	m_levels = 0;
	m_depth=-1;
        m_root = buildOctreeNode(0, std::make_pair(0, m_pnts.cols()), bbox);
	m_depth=m_levels;


        std::cout << "building the nodes up to level"<<m_depth << std::endl;        	
	//m_root->print();

	
    }

    void printInteractionList(std::shared_ptr<OctreeNode>  src) {
	std::cout<<"interactions for "<<src->id()<<" "<<src->level()<<std::endl;
	for( auto near : src->nearTargets()) {
	    std::cout<<near.first<<" "<<near.second<<std::endl;
	}
	std::cout<<"far"<<std::endl;

	for( auto far : src->farTargets()) {
	    std::cout<<far.first<<" "<<far.second<<std::endl;
	}

		    
    }

    void buildInteractionList(const Octree&  target_tree)
    {
	buildInteractionList(m_root,target_tree.m_root);
    }

    void buildInteractionList(std::shared_ptr<OctreeNode>  src,std::shared_ptr<const OctreeNode>  target)
    {
	if(!src || !target || !src->hasPoints() || ! target->hasPoints()) {
	    return;
	}


	//Ideally, this interaction is admissible, so we can interpolate it at this level
	if(m_isAdmissible(src->boundingBox(),target->boundingBox())) {
	    src->addFarInteraction(*target);
	    return;
	}

	
	//If both of them are leaves, we can't proceed by recursion. So let's just stop and
	//do it the hard way	
	if(src->isLeaf() && target->isLeaf()) {	    
	    src->addNearInteraction(*target);
	    return;
	}

	//If either src or target has children recurse down
	if(src->isLeaf() && !target->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src,target->child(j));
	    }
	    return;
	}

	if(target->isLeaf() && !src->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j),target);
	    }
	    return;
	}

	//if both src and target have children, we recurse down the one with the larger bbox
	if(src->boundingBox().sideLength() > target->boundingBox().sideLength()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j),target);
	    }
	    return;
	}else {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src,target->child(j));
	    }
	    return;
	}

	std::cout<<"this does not happen!"<<std::endl;
	src->print();
	target->print();
    }

    void calculateInterpolationRange(  std::function<Eigen::Vector<int,DIM>(double,int)> order_for_H,std::function<Eigen::Vector<size_t,DIM>(double,int )> N_for_H, const Octree& target)
    {
	BoundingBox<DIM> global_box;
	//global_box.min().fill(10);
	//global_box.max().fill(0);

#ifdef BE_FAST
	TransformationMode mode=Decomposition;
#else
	TransformationMode mode=TwoGrid;
#endif

#ifdef RECURSIVE_MULT
	
	int min_boxes=32;
	int min_recursive_level=0;
        for(;min_recursive_level<levels();min_recursive_level++) {
            if(numBoxes(min_recursive_level) > min_boxes) {
                break;
            }
        }

	
#else
	const size_t min_recursive_level = levels();
#endif

	std::cout<<"min rec"<<min_recursive_level;
	

	const auto & target_points=target.points();
	tbb::queuing_mutex activeConeMutex;

	std::vector<tbb::spin_mutex> target_mutexes(target.numPoints());
	Eigen::Vector<size_t,DIM> oldN;
	//No interpolation at the two highest levels 
	for (size_t level=0;level<levels();level++) {
	    std::cout<<"l= "<<level<<std::endl;
	    //update all nodes in this level

	    std::array<std::vector<ConeRef>,2> activeCones;
	    std::vector<ConeRef> leafCones;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
            for(size_t n=r.begin();n<r.end();++n) {
		std::shared_ptr<OctreeNode> node=m_nodes[level][n];
		BoundingBox<DIM> box;


		//do nothing  if there are no sources
		if(node==0 || node->pntRange().first==node->pntRange().second)
		    continue;

		const Point xc=node->boundingBox().center();
		
		const double H=node->boundingBox().sideLength();		
		const std::vector<IndexRange> farTargets=node->farTargets();
#ifdef EXACT_INTERP_RANGE

		
		for(const IndexRange& iR : farTargets)
		{
		    for(int i=iR.first;i<iR.second;i++) {
			//std::cout<<"asd"<<node->boundingBox().exteriorDistance(m_targets.col(i))<<" "<<H<<std::endl;
			
			const auto s=Util::cartToInterp<DIM>(target_points.col(i),xc,H);
			assert(s[0]<=sqrt(DIM)/DIM);
			
			box.extend(s); //make sure the target is in the interpolation domain
		    }
		}
		
		BoundingBox<DIM> pBox;
		std::shared_ptr<const OctreeNode> parent=node->parent().lock();
		//now also add all the parents targets	       
		if(parent && parentHasFarTargets(node))
		{	    
		    pBox=parent->interpolationRange();
		    const Point pxc=parent->boundingBox().center();
		    const double pH=parent->boundingBox().sideLength();


		    if(!pBox.isNull())
		    {
			//transform the parents interpolation range to the physical coordinates
			auto cMin=Util::interpToCart<DIM>(pBox.min().array(),pxc,pH);
			auto cMax=Util::interpToCart<DIM>(pBox.max().array(),pxc,pH);


			//pull those physical coordinates back to the interpolation-coordinates of node
			//box.extend(Util::cartToInterp<DIM>(cMin,xc,H));	
			//box.extend(Util::cartToInterp<DIM>(cMax,xc,H));

			//TODO check if this is necessary
			const ConeDomain<DIM>& p_grid=parent->coneDomain();
			auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH,0));
			for(size_t el : p_grid.activeCones() ) {			
			    for (size_t i=0;i<chebNodes.cols();i++)				auto pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)).array(),pxc,pH);
				box.extend(Util::cartToInterp<DIM>(pnt,xc,H).matrix());
			    }
			} 
		    }
		}

		if(!box.isNull()) {
		    //extend the box slighlty such that the boundary points are not used for interpolation
		    box.extend((box.min().array()-1e-06).matrix());
		    box.extend((box.max().array()+1e-06).matrix());
		}
#else

		//just use a default value for the boxes

		double smax=sqrt(DIM)/DIM;
		//if the sources and targets are well-separated we don't have to cover the near field 
		const double dist=0;//bbox(0,0).exteriorDistance(target.bbox(0,0));
		if(dist >0) {
		    smax=std::min(smax, H/dist);
		}

                
		std::shared_ptr<const OctreeNode> parent=node->parent().lock();
		BoundingBox<DIM> pBox;
		//now also add all the parents targets	       
		if(parent && parentHasFarTargets(node))
		{	    
		    pBox=parent->interpolationRange();
		}

		
                const double dist_t=0;//target.bbox(0,0).exteriorDistance(xc);
		double smin=0.5*H/m_diameter;///(dist_t+target.m_diameter+m_diameter);
                if(!pBox.isNull()) {
		    //std::cout<<"diam:"<<m_diameter<<" "<<pBox.min()[0]<<std::endl;
                    smin=std::min(smin,0.75/(sqrt(DIM)/3.0+2.0/pBox.min()[0]));
		    //smin=std::min(smin, pBox.min()[0]/2.0);
		}
		smin=1e-3;
                //1e-3;//H/(m_diameter+dist+target.m_diameter);

		
		box.min()(0)=smin;
		box.max()(0)=smax;

			
		if constexpr(DIM==2) {	    
		    box.min()(1)=-M_PI;
		    box.max()(1)=M_PI;
		}else{
		    box.min()(1)=0;
		    box.max()(1)=M_PI;
			    
		    box.min()(2)=-M_PI;
		    box.max()(2)=M_PI;
		}
	
		    
		
#endif		

		BoundingBox<DIM> tbox(pBox);

		
		ConeDomain<DIM> domain(N_for_H(H,0),box);
		ConeDomain<DIM> trans_domain(N_for_H(2*H,0),tbox);

		auto hoN=N_for_H(H,1);

		ConeDomain<DIM> coarseDomain(hoN,box);

		    
		assert(H>0);
		if(N_for_H(H,0)!=oldN) {
		    oldN=N_for_H(H,0);
		    std::cout<<"n="<<N_for_H(H,0).transpose()<<"/"<<N_for_H(H,1).transpose()<<std::endl;
		    std::cout<<"box="<<box<<" "<<box.isNull()<<"H="<<H<<std::endl;
		}

		if(!box.isNull())
		    global_box.extend(box);

		//now we need to do the whole thing again to figure out which cones are active...
		// 0: where do i need to compute the points directly/from the children in order to be able to sample FF and first rotation
		// 1: where do i need to compute the points using  the first rotation in order to be able to compute the translation
		// 2: where do i need to compute the points using the translation in order to be able to compute the second rotation
		std::array<std::unordered_set<size_t>,2> is_cone_active; 
		


		for(const IndexRange& iR : farTargets)
		{				    
		    for(int i=iR.first;i<iR.second;i++)
		    {
			const auto s=Util::cartToInterp<DIM>(target_points.col(i),xc,H);			  
			auto coneId=domain.elementForPoint(s);
			is_cone_active[0].insert(coneId);
		    }
		}

		
		//now also add all the parents targets
		if(!pBox.isNull())
		{
		    const Point pxc=parent->boundingBox().center();
		    const double pH=parent->boundingBox().sideLength();

		    const ConeDomain<DIM>& p_grid=parent->coneDomain();
		    auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH,0));
		    
		    
		    if(mode==Decomposition){
			//now we add all the points used in the point-and-shoot method. Second rotation:
			for(size_t el : p_grid.activeCones() ) {
			    const auto direction=xc-pxc;

			    auto points=p_grid.rotated_points(el,chebNodes,direction, false);

			    for (size_t i=0;i<points.cols();i++) {
				auto coneId=trans_domain.elementForPoint(points.col(i));
				is_cone_active[2].insert(coneId);
			    }
			}


			//translation
			for(size_t el=0;el<trans_domain.n_elements();el++) {
			    if(is_cone_active[2].count(el)>0) {			    
				auto points=trans_domain.translated_points(el,chebNodes,xc,H,pxc,pH);

				for (size_t i=0;i<points.cols();i++) {
				    auto coneId=domain.elementForPoint(points.col(i));
				    is_cone_active[1].insert(coneId);
				}
			    }
			}


			//first rotation
			for(size_t el=0;el<domain.n_elements();el++) {
			    if(is_cone_active[1].count(el)>0) {
				const auto direction=xc-pxc;
				auto points=domain.rotated_points(el,chebNodes,direction, true);

				for (size_t i=0;i<points.cols();i++) {
				    auto coneId=domain.elementForPoint(points.col(i));
				    is_cone_active[0].insert(coneId);
				}
			    }
			}
		    }
		    else if(mode==TwoGrid) {
			//We use a coarse grid with high order and a fine grid of lower order
			auto HoChebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH,1));			
			const ConeDomain<DIM>& p_hoGrid=parent->coneDomain(1);
			//

			PointArray pnts(DIM,HoChebNodes.cols());
			PointArray interp_pnts(DIM,HoChebNodes.cols());
			for(size_t el : p_hoGrid.activeCones() ) {
			    const auto direction=xc-pxc;
			    pnts=Util::interpToCart<DIM>(p_hoGrid.transform(el,HoChebNodes).array(),pxc,pH);
			    Util::cartToInterp2<DIM>(pnts.array(),xc,H,interp_pnts.array());
			    for (size_t i=0;i<HoChebNodes.cols();i++) {
				auto coneId=domain.elementForPoint(interp_pnts.col(i));				
				is_cone_active[0].insert(coneId);
			    }
			}			

		    }
		    else   {
			//now we add all the points used in the regular method
			for(size_t el : p_grid.activeCones() ) {
	    
			    for (size_t i=0;i<chebNodes.cols();i++) {
				Point cart_pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)).array(),pxc,pH);
				Point interp_pnt=Util::cartToInterp<DIM>(cart_pnt,xc,H);

				auto coneId=domain.elementForPoint(interp_pnt);
				is_cone_active[0].insert(coneId);
			    }
			}
		    }
		}


		if(mode==TwoGrid) {
		    auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(H,1));
		    //reinterpolation step from HO to regular
		    PointArray interp_pnts(DIM, chebNodes.cols());
		    for(size_t el=0;el<domain.n_elements();el++) {
			if(is_cone_active[0].count(el)>0) {
			    interp_pnts=domain.transform(el,chebNodes).array();				
			    for (size_t i=0;i<chebNodes.cols();i++) {
				
				auto coneId=coarseDomain.elementForPoint(interp_pnts.col(i));				
				is_cone_active[1].insert(coneId);
			    }
			    
			}
		    }
		}
		
		for (int step=0;step<2;step++ ) {
		     std::vector<size_t> local_active_cones;
		     std::unordered_map<size_t,size_t> cone_map;
		     //local_active_cones.reserve(domain.n_elements());
		     for( size_t i : is_cone_active[step])
		     {			 
			     ConeRef cone(level, i, local_active_cones.size(), n);
			     tbb::queuing_mutex::scoped_lock lock(activeConeMutex);
			     activeCones[step].push_back(cone);

			     int leafStep=0;
			     if(mode==TwoGrid) {
				 leafStep=1;
			     }
			     
			     if(node->isLeaf() && step==leafStep) {
				 leafCones.push_back(cone);
			     }
			     local_active_cones.push_back(i);
			     cone_map[i]=local_active_cones.size()-1;
			 
		     }

		     //local_active_cones.trim();
		     ConeDomain d0=domain;
		     if(step==2 && mode==Decomposition) {
			 d0=trans_domain;
		     }
		     if(step==1 && mode==TwoGrid) {
			 d0=coarseDomain;
		     }
		     d0.setActiveCones(local_active_cones);		
		     d0.setConeMap(cone_map);		     

		     //std::cout<<"level"<<node->level()<<"range:"<<box.min().transpose()<<" "<<box.max().transpose()<<std::endl;
		     node->setConeDomain(d0,step);		     
		}
		//node->setInterpolationRange(box);
	    }});
	    {
		tbb::queuing_mutex::scoped_lock lock(activeConeMutex);
		std::cout<<"level= "<<level<<" active cones"<<activeCones[0].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,0).n_elements()<<std::endl;
		std::cout<<"level= "<<level<<" active cones"<<activeCones[1].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,1).n_elements()<<std::endl;		
		std::cout<<"level= "<<level<<" leafCones "<<leafCones.size()<<std::endl;
	    }
	    m_activeCones.push_back(activeCones);
	    m_leafCones.push_back(leafCones);


	    if(level< min_recursive_level) {
		//compute the farFieldBoxes
		std::vector<std::vector<size_t> > ffB;
		ffB.reserve(target.points().size());
		for(size_t l=0;l<target.points().size();l++){		
		    ffB.push_back(std::vector<size_t>());
		}

		tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
		    for(size_t n=r.begin();n<r.end();++n) {
			std::shared_ptr<OctreeNode> node=m_nodes[level][n];
			const std::vector<IndexRange> farTargets=node->farTargets();
			for( const auto tRange : farTargets) {
			    for(size_t trg=tRange.first;trg<tRange.second;++trg) {
				tbb::spin_mutex::scoped_lock lock(target_mutexes[trg]);
				ffB[trg].push_back(n);
			    }
			}
		    }
		});
		m_farFieldBoxes.push_back(ffB);
	    }else {
		m_farFieldBoxes.push_back(std::vector<std::vector<size_t> >());
	    }

	}

	std::cout<<"interp_domain:" <<global_box<<std::endl;

    }


	
    
    inline double diameter () const
    {
	return m_diameter;
    }

    unsigned int levels() const
    {
        return m_levels;
    }

    long int child(size_t level, size_t id, size_t childIndex) const
    {
	const auto  child=m_nodes[level][id]->child(childIndex);
	if(child) {
	    return m_nodes[level][id]->child(childIndex)->id();
	}else{
	    return -1;
	}
    }

    const std::vector<size_t> activeChildren( size_t level,size_t id) const
    {
        const auto node=m_nodes[level][id];
        std::vector<size_t> aC;
        aC.reserve(N_Children);
        for (int i=0;i<N_Children;i++) {
            const auto child=node->child(i);
            if(child->hasPoints()) {
                aC.push_back(child->id());
            }
        }
        return aC;
    }

    unsigned int numBoxes(unsigned int level) const
    {
        return m_numBoxes[level];
    }

    const auto points(IndexRange index) const
    {
        return m_pnts.middleCols(index.first, index.second - index.first);
    }

    const auto point(size_t id) const
    {
        return m_pnts.col(id);
    }


    const std::vector<size_t> permutation() const
    {
        return m_permutation;
    }




    const IndexRange points(unsigned int level, size_t i) const
    {
	assert(level< m_nodes.size());
	assert(i < m_nodes[level].size());
	std::shared_ptr<OctreeNode> node = m_nodes[level][i];

        return node->pntRange();
    }

    inline const std::vector<IndexRange> nearTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return node->nearTargets();
    }

    inline const std::vector<IndexRange> farTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return node->farTargets();
    }


    bool hasFarTargetsIncludingAncestors(unsigned int level, size_t i) const
    {
	const std::shared_ptr<const OctreeNode>& node = m_nodes[level][i];
	if(node) {	    
	    return node->farTargets().size()>0 ||parentHasFarTargets(node);
	}
	return false;
    }

    bool parentHasFarTargets(const std::shared_ptr<const OctreeNode>& node) const
    {
	const std::shared_ptr<const OctreeNode>& parent = node->parent().lock();
	if(parent) {
	    return parent->farTargets().size()>0 || parentHasFarTargets(parent);
	}else{
	    return false;
	}
    }

    bool parentHasFarTargets(unsigned int level, size_t i) const
    {
	const std::shared_ptr<const OctreeNode>& node = m_nodes[level][i];
	if(node) {
	    return parentHasFarTargets(node);
	}
	return false;
    }

    const BoundingBox<DIM> bbox(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->boundingBox();
    }


    const BoundingBox<DIM> interpolationRange(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->interpolationRange();
    }

    const ConeDomain<DIM> coneDomain(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->coneDomain();
    }

    const ConeDomain<DIM> coneDomain(unsigned int level, size_t i,size_t substep) const
    {
        return m_nodes[level][i]->coneDomain(substep);
    }



    const std::vector<size_t> childBoxes(unsigned int level, size_t i) const
    {
	auto parent=m_nodes[level][i];
	std::vector<size_t> children;
	for(int i=0;i<N_Children;i++) {
	    const auto child=parent->child(i);
	    if(child) {
		children.push_back(child->id());
	    }	    
	}
	return children;
    }
    
    const size_t parentId(unsigned int level, size_t i) const
    {
	auto parent=m_nodes[level][i]->parent().lock();
	size_t id = parent->id();
	assert(m_nodes[level-1][id]==parent);
        return id;
    }

    const bool hasPoints(unsigned int level, size_t i) const
    {
        const auto range = points(level, i);
        return range.first != range.second;
    }

    const bool isLeaf(unsigned int level, size_t i) const
    {
	return m_nodes[level][i]->isLeaf();        
    }

    size_t numActiveCones(size_t level,size_t step=0) const {
	return m_activeCones[level][step].size();
    }

    ConeRef activeCone(size_t level,size_t id,size_t step=0) const
    {
	return m_activeCones[level][step][id];
    }


    size_t numLeafCones(size_t level) const
    {
	return m_leafCones[level].size();
    }

    ConeRef leafCone(size_t level, size_t num) const
    {
	return m_leafCones[level][num];
    }


    const std::vector<size_t>& farfieldBoxes(size_t level, size_t targetPoint) const {
	return  m_farFieldBoxes[level][targetPoint];
    }

    size_t numPoints() const {
	return m_pnts.size();
    }

    void sanitize()
    {
	Eigen::VectorXi leaf_indices(m_pnts.cols());	
	leaf_indices.fill(0);
        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_pnts.cols());
	    indices=leaf_indices;            
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->pntRange().first;
                size_t b = m_nodes[level][i]->pntRange().second;
		
                for (int l = a; l < b; l++) {
                    indices[l] += 1;
		    if(m_nodes[level][i]->isLeaf()) {
			leaf_indices[l]+=1;
		    }
		    
                }
            }

            for (int i = 0; i < indices.size(); i++) {
                if (indices[i] != 1) {
                    std::cout << "wrong" << indices[i] << " " << i << " level" << level << std::endl;
                    std::cout << " at " << m_pnts.col(i) << std::endl;
                }
                assert(indices[i] == 1);
            }
        }
    }

private:
    std::shared_ptr<OctreeNode> buildOctreeNode(std::shared_ptr<OctreeNode > parent, const IndexRange &pnt_range, const BoundingBox<DIM> &bbox, unsigned int level = 0)
    {
	
	if(pnt_range.first==pnt_range.second) //we only keep non-empty nodes around
	{
	    return 0;
	}
	
        if (level >= m_levels) {
            m_levels++;
            m_nodes.push_back(std::vector<std::shared_ptr<OctreeNode> >());
            m_numBoxes.push_back(0);
        }

	auto node = std::make_shared<OctreeNode >(parent, level);

	
	node->setPntRange(pnt_range);
	    
	node->setBoundingBox(bbox);

	m_numBoxes[level] += 1;
	node->setId(m_nodes[level].size());
	
	m_nodes[level].push_back(node);
	

	//check how big we are. If the number of points
	//is small enough we create a new leaf.
	const size_t N= pnt_range.second-pnt_range.first;

	
	if( N<= m_maxLeafSize ) {
	    return node;
	}

        size_t pnt_idx = pnt_range.first;

        for (int j = 0; j < N_Children; j++) {
            Point min;
            Point max;

            //std::cout<<"finding bbox from parent"<<std::endl;
            min = bbox.min();
            max = bbox.max();

            Eigen::Vector<double, DIM> size = 0.5 * bbox.diagonal();

            //find the quadrant that the next src idx belongs to

            auto tuple_idx = compute_tuple_idx(j);
            min.array() += size.array() * tuple_idx.array();
            max = min + size;

            BoundingBox<DIM> child_bbox(min, max);

            //std::cout<<"building child "<<j<<" at"<<child_bbox.min().transpose()<<" "<<child_bbox.max().transpose()<<std::endl;

            size_t end_pnt = pnt_idx;

            while (end_pnt < pnt_range.second && child_bbox.contains(m_pnts.col(end_pnt).matrix())) {
                ++end_pnt;
            }


            //assert(src_idx>=src_range.first && end_src <=src_range.second);
            //std::cout<<"src:"<<src_idx<<end_src<<std::endl;
            const IndexRange pnts(pnt_idx, end_pnt);
            node->setChild(j, buildOctreeNode(node, pnts, child_bbox, level + 1));

            pnt_idx = end_pnt;
        }

        return node;

    }

    inline Eigen::Vector<double, DIM> compute_tuple_idx(size_t idx) const
    {
        Eigen::Vector<double, DIM> tuple;
        tuple.fill(0);

        for (size_t j = 0; j < DIM; j++) {
            tuple[j] = idx & 1;

            idx = idx >> 1;

        }

        return tuple;

    }

    const auto& points() const {
	return m_pnts;
    }


private:
    std::shared_ptr<OctreeNode > m_root;
    std::vector<std::vector<std::shared_ptr<OctreeNode> > > m_nodes;
    std::vector<std::array<std::vector<ConeRef>,2> > m_activeCones;
    std::vector<std::vector<ConeRef> > m_leafCones;
    std::vector<std::vector<std::vector<size_t> > > m_farFieldBoxes;  // on each level: for each target point y store the source boxes such that y is in the farfield 
    std::vector<unsigned int> m_numBoxes;
    unsigned int m_levels;

    unsigned int m_depth;
    size_t m_maxLeafSize;
    PointArray m_pnts;
    std::vector<size_t> m_permutation;
    double m_diameter;

    std::function<bool(const BoundingBox<DIM>&, const BoundingBox<DIM>&) > m_isAdmissible;

};

#endif
