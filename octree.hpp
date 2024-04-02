#ifndef _OCTREE_HPP_
#define _OCTREE_HPP_

#include <Eigen/Dense>
#include <memory>
#include <map>
#include <vector>
#include <execution>
#include <iostream>

#include "util.hpp"
#include "boundingbox.hpp"
#include "zorder_less.hpp"
#include "chebinterp.hpp"
#include "cone_domain.hpp"

#include <tbb/spin_mutex.h>
typedef std::pair<size_t, size_t> IndexRange;


template<typename T, size_t DIM>
class Octree
{

public:
    enum {N_Children = 1 << DIM };
    typedef Eigen::Matrix<double, DIM, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<double, DIM> Point;

    class OctreeNode
    {
    private:
        std::shared_ptr<OctreeNode > m_parent;
        long int m_id;
        std::shared_ptr<OctreeNode> m_children[N_Children];
        //std::vector<std::shared_ptr<const OctreeNode> > m_neighbors;

	std::vector<IndexRange> m_farTargets;
	std::vector<IndexRange> m_nearTargets;

        IndexRange m_srcRange;
        IndexRange m_targetRange;

	ConeDomain<DIM> m_coneDomain;

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

        IndexRange srcRange() const
        {
            return m_srcRange;
        }

        IndexRange targetRange() const
        {
            return m_targetRange;
        }


        void setSrcRange(const IndexRange &range)
        {
            m_srcRange = range;

        }

        void setTargetRange(const IndexRange &range)
        {
            m_targetRange = range;
        }

        void setChild(size_t idx, std::shared_ptr<OctreeNode> node)
        {
	    if(node!=0) {
		m_children[idx] = node;
		m_isLeaf = false;
	    }
        }

	void setConeDomain(const ConeDomain<DIM>& domain)
	{
	    m_coneDomain=domain;
	}

	const ConeDomain<DIM>& coneDomain() const
	{
	    return m_coneDomain;
	}


	BoundingBox<DIM> interpolationRange() const
	{
	    return m_coneDomain.domain();
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

        bool hasSources() const
        {            
            return m_srcRange.first != m_srcRange.second;
        }

        unsigned int level() const
        {
            return m_level;
        }

	bool hasTargets() const
	{
	    return  m_targetRange.first !=m_targetRange.second;
	}

        const std::shared_ptr<const OctreeNode> parent() const
        {
            return m_parent;
        }

	void addNearInteraction(const OctreeNode& target)
	{
	    if(target.hasTargets())
		m_nearTargets.push_back(target.targetRange());
	}
	
	void addFarInteraction(const OctreeNode& target)
	{
	    if(target.hasTargets())
		m_farTargets.push_back(target.targetRange());
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
            std::cout << "(" << m_srcRange.first << " " << m_srcRange.second << ")"; //<<std::endl;
            //std::cout << m_bbox.min().transpose() << "   " << m_bbox.max().transpose() << std::endl;
	    std::cout<< m_isLeaf <<std::endl;

            //std::cout<<"("<<m_srcRange.first<<" "<<m_srcRange.second<<")"<<std::endl;
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

	double eta=(double) sqrt((double) DIM);;
	m_isAdmissible= [eta] (const BoundingBox<DIM>& src,const BoundingBox<DIM>& target) { return target.exteriorDistance(src.center()) >= eta* src.sideLength();};

    }

    ~Octree()
    {

    }

    void build(const PointArray &srcs, const PointArray &targets)
    {
        std::cout << "building a new octree" << srcs.cols() << ", " << targets.cols() << std::endl;

        std::cout << "finding bbox" << std::endl;
        Point min = srcs.col(0);
        Point max = srcs.col(0);

        for (size_t i = 0; i < srcs.cols(); i++) {
            min = min.array().min(srcs.col(i).array());
            max = max.array().max(srcs.col(i).array());
        }

        for (size_t i = 0; i < targets.cols(); i++) {
            min = min.array().min(targets.col(i).array());
            max = max.array().max(targets.col(i).array());
        }

        //make the bbox slightly larger to not have to deal with boundary issues
        min.array() -= 1e-8 * min.norm();
        max.array() += 1e-8 * max.norm();

        BoundingBox<DIM> bbox(min, max);
        std::cout << "bbox=" << min.transpose() << "\t" << max.transpose() << std::endl;

        //sort the points by their morton order for better locality later on
        std::cout << "sorting..." << std::endl;
        m_src_permutation = Util::sort_with_permutation(std::execution::par, srcs.colwise().begin(), srcs.colwise().end(), zorder_knn::Less<Point, DIM>(bbox));
        m_srcs = Util::copy_with_permutation(srcs, m_src_permutation);

        m_target_permutation = Util::sort_with_permutation(std::execution::par, targets.colwise().begin(), targets.colwise().end(), zorder_knn::Less<Point, DIM>(bbox));
        m_targets = Util::copy_with_permutation(targets, m_target_permutation);


	m_diameter=bbox.diagonal().norm();

	m_levels = 0;
	m_depth=-1;
        m_root = buildOctreeNode(0, std::make_pair(0, m_srcs.cols()), std::make_pair(0, m_targets.cols()), bbox);
	m_depth=m_levels;


        std::cout << "building the nodes up to level"<<m_depth << std::endl;
        
	
        buildInteractionList(m_root,m_root);
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

    void buildInteractionList(std::shared_ptr<OctreeNode>  src,std::shared_ptr<OctreeNode>  target)
    {
	if(!src || !target || !src->hasSources() || ! target->hasTargets()) {
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

    void calculateInterpolationRange(  std::function<Eigen::Vector<int,DIM>(double)> order_for_H,std::function<Eigen::Vector<size_t,DIM>(double)> N_for_H)
    {
	BoundingBox<DIM> global_box;
	global_box.min().fill(0);
	global_box.max().fill(0);
	
	Eigen::Vector<size_t,DIM> oldN;
	//No interpolation at the two highest levels 
	for (size_t level=0;level<levels();level++) {
	    //update all nodes in this level
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
            for(size_t n=r.begin();n<r.end();++n) {
		std::shared_ptr<OctreeNode> node=m_nodes[level][n];
		BoundingBox<DIM> box;


		//do nothing  if there are no sources
		if(node==0 || node->srcRange().first==node->srcRange().second)
		    continue;


		const Point xc=node->boundingBox().center();
		const double H=node->boundingBox().sideLength();
		const std::vector<IndexRange> farTargets=node->farTargets();
		
		for(const IndexRange& iR : farTargets)
		{
		    for(int i=iR.first;i<iR.second;i++) {
			//std::cout<<"asd"<<node->boundingBox().exteriorDistance(m_targets.col(i))<<" "<<H<<std::endl;
			
			const auto s=Util::cartToInterp<DIM>(m_targets.col(i),xc,H);
			assert(s[0]<=sqrt(DIM)/DIM);
			
			box.extend(s); //make sure the target is in the interpolation domain
		    }
		}

		BoundingBox<DIM> pBox;
		//now also add all the parents targets	       
		if(node->parent() && parentHasFarTargets(node))
		{
		    
		    pBox=node->parent()->interpolationRange();
		    const Point pxc=node->parent()->boundingBox().center();
		    const double pH=node->parent()->boundingBox().sideLength();


		    if(!pBox.isNull())
		    {
			//transform the parents interpolation range to the physical coordinates
			auto cMin=Util::interpToCart<DIM>(pBox.min().array(),pxc,pH);
			auto cMax=Util::interpToCart<DIM>(pBox.max().array(),pxc,pH);


			//pull those physical coordinates back to the interpolation-coordinates of node
			//box.extend(Util::cartToInterp<DIM>(cMin,xc,H));	
			//box.extend(Util::cartToInterp<DIM>(cMax,xc,H));

			//TODO check if this is necessary
			const ConeDomain<DIM>& p_grid=node->parent()->coneDomain();
			auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH));
			for(size_t el : p_grid.activeCones() ) {			
			    for (size_t i=0;i<chebNodes.cols();i++) {
				auto pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)).array(),pxc,pH);
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
		

		ConeDomain<DIM> domain(N_for_H(H),box);
		if(N_for_H(H)!=oldN) {
		    oldN=N_for_H(H);
		    std::cout<<"n="<<N_for_H(H).transpose()<<std::endl;
		    std::cout<<"box="<<box<<" "<<box.isNull()<<std::endl;
		}

		if(!box.isNull())
		    global_box.extend(box);

		//now we need to do the whole thing again to figure out which cones are active...
		std::vector<bool> is_cone_active(domain.n_elements());
		std::fill(is_cone_active.begin(),is_cone_active.end(),false);
		/*for (int i=0;i<domain.n_elements();i++) {
		    assert(is_cone_active[i]==0);
		    }*/
		
		//now also add all the parents targets		
		for(const IndexRange& iR : farTargets)
		{				    
		    for(int i=iR.first;i<iR.second;i++)
		    {
			const auto s=Util::cartToInterp<DIM>(m_targets.col(i),xc,H);			  
			
			auto coneId=domain.elementForPoint(s);
			is_cone_active[coneId]=true;
		    }
		}

		if(!pBox.isNull())
		{
		    const Point pxc=node->parent()->boundingBox().center();
		    const double pH=node->parent()->boundingBox().sideLength();

		    const ConeDomain<DIM>& p_grid=node->parent()->coneDomain();		    
		    auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH));

		    for(size_t el : p_grid.activeCones() ) {			
		    	for (size_t i=0;i<chebNodes.cols();i++) {
			    auto cart_pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)).array(),pxc,pH);
			    auto interp_pnt=Util::cartToInterp<DIM>(cart_pnt,xc,H);
			    auto coneId=domain.elementForPoint(interp_pnt);
			    is_cone_active[coneId]=true;
			}
		    }
		}


		std::vector<size_t> active_cones;
		std::vector<size_t> cone_map(domain.n_elements());
		for(size_t i=0;i<domain.n_elements();i++)
		{
		    if(is_cone_active[i]) {
			active_cones.push_back(i);
			cone_map[i]=active_cones.size()-1;
		    }
		    
		}

		//std::cout<<"active="<<(100*active_cones.size())/domain.n_elements()<<std::endl;
		domain.setActiveCones(active_cones);
		domain.setConeMap(cone_map);

		//std::cout<<"level"<<node->level()<<"range:"<<box.min().transpose()<<" "<<box.max().transpose()<<std::endl;
		node->setConeDomain(domain);
		//node->setInterpolationRange(box);
	    }});
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
            if(child->hasSources()) {
                aC.push_back(child->id());
            }
        }
        return aC;
    }

    unsigned int numBoxes(unsigned int level) const
    {
        return m_numBoxes[level];
    }

    const auto sourcePoints(IndexRange index) const
    {
        return m_srcs.middleCols(index.first, index.second - index.first);
    }

    const auto targetPoints(IndexRange index) const
    {
        return m_targets.middleCols(index.first, index.second - index.first);
    }

    const std::vector<size_t> src_permutation() const
    {
        return m_src_permutation;
    }

    const std::vector<size_t>  target_permutation() const
    {
        return m_target_permutation;
    }

    const IndexRange sources(unsigned int level, size_t i) const
    {
	assert(level< m_nodes.size());
	assert(i < m_nodes[level].size());
	std::shared_ptr<OctreeNode> node = m_nodes[level][i];

        return node->srcRange();
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

    bool parentHasFarTargets(const std::shared_ptr<const OctreeNode>& node) const
    {
	const std::shared_ptr<const OctreeNode>& parent = node->parent();
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

    /*    const IndexRange siblingSources(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        IndexRange srcs = node->srcRange();
        for (int j = 0; j < Octree::N_Children; j++) {
            const std::shared_ptr<const OctreeNode > sibling = node->parent()->child(j);
            if (sibling && node->boundingBox().squaredExteriorDistance(sibling->boundingBox()) <= std::numeric_limits<double>::epsilon()) {
                srcs.first = std::min(srcs.first, sibling->srcRange().first);
                srcs.second = std::max(srcs.second, sibling->srcRange().second);
            }
        }

        return srcs;
    }

    inline const IndexRange siblingTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return siblingTargets(*node);
    }

    inline const IndexRange siblingTargets(const OctreeNode &node) const
    {
        IndexRange targets = node.parent()->targetRange();

        return targets;
    }

    inline const std::vector<IndexRange> neighborTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return neighborTargets(*node);
    }

    const std::vector<IndexRange> neighborTargets(const OctreeNode &node) const
    {
        std::vector<IndexRange> neighbors;

        for (auto nb : node.neighbors()) {
            neighbors.push_back(nb->targetRange());
        }
        return neighbors;

    }
    const std::vector<IndexRange> cousinTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
	assert(node->id()>=0);
	return cousinTargets(node);
    }

    const std::vector<IndexRange> cousinTargets(std::shared_ptr<OctreeNode> node) const
    {        
        std::vector<IndexRange> cousins;

	const double H=node->boundingBox().sideLength();
        //add some of the children of the parents neighbours
        for (auto p_nb : node->parent()->neighbors()) {
            //add all the (possible) cousins which have positive distance to the current node (i.e., not neighbors)
            for (int l = 0; l < N_Children; l++) {
                const std::shared_ptr<const OctreeNode > cousin = p_nb->child(l);
                if (cousin && node->boundingBox().squaredExteriorDistance(cousin->boundingBox())>H*H)
		{
                    cousins.push_back(cousin->targetRange());
                }
            }

        }

        return cousins;

    }
    */
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


    
    const size_t parentId(unsigned int level, size_t i) const
    {
	size_t id = m_nodes[level][i]->parent()->id();
	assert(m_nodes[level-1][id]==m_nodes[level][i]->parent());
        return id;
    }

    const bool hasSources(unsigned int level, size_t i) const
    {
        const auto range = sources(level, i);
        return range.first != range.second;
    }

    const bool isLeaf(unsigned int level, size_t i) const
    {
	return m_nodes[level][i]->isLeaf();        
    }


    void sanitize()
    {
	Eigen::VectorXi leaf_indices(m_srcs.cols());	
	leaf_indices.fill(0);
        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_srcs.cols());
	    indices=leaf_indices;            
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->srcRange().first;
                size_t b = m_nodes[level][i]->srcRange().second;
		
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
                    std::cout << " at " << m_srcs.col(i) << std::endl;
                }
                assert(indices[i] == 1);
            }
        }

	leaf_indices.fill(0);
        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_targets.cols());	    
            indices=leaf_indices;
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->targetRange().first;
                size_t b = m_nodes[level][i]->targetRange().second;

                for (int l = a; l < b; l++) {
                    indices[l] += 1;
		    if(m_nodes[level][i]->isLeaf()) {
			leaf_indices[l]+=1;
		    }
                }
            }

            for (int i = 0; i < indices.size(); i++) {
                if (indices[i] != 1) {
                    std::cout << "wrong22" << indices[i] << " " << i << std::endl;
                }
                assert(indices[i] == 1);
            }
        }

    }

private:
    std::shared_ptr<OctreeNode> buildOctreeNode(std::shared_ptr<OctreeNode > parent, const IndexRange &src_range, const IndexRange &target_range, const BoundingBox<DIM> &bbox, unsigned int level = 0)
    {
	
	if(src_range.first==src_range.second && target_range.first==target_range.second) //we only keep non-empty nodes around
	{
	    return 0;
	}
	
        if (level >= m_levels) {
            m_levels++;
            m_nodes.push_back(std::vector<std::shared_ptr<OctreeNode> >());
            m_numBoxes.push_back(0);
        }

	auto node = std::make_shared<OctreeNode >(parent, level);

	for(size_t i=target_range.first;i<target_range.second;i++) {
	    assert(bbox.contains(m_targets.col(i)));
	}

	
	node->setSrcRange(src_range);
	node->setTargetRange(target_range);
	    
	node->setBoundingBox(bbox);

	m_numBoxes[level] += 1;
	node->setId(m_nodes[level].size());
	
	m_nodes[level].push_back(node);
	


	//check how big we are. If the number of sources and targets
	//is small enough we create a new leaf.
	const size_t N= std::max(src_range.second-src_range.first,
				 target_range.second-target_range.first);

	
	if( N<= m_maxLeafSize ) {
	    return node;
	}

        size_t src_idx = src_range.first;
        size_t target_idx = target_range.first;

        for (int j = 0; j < N_Children; j++) {
            Point min;
            Point max;

            //std::cout<<"finding bbox from parent"<<std::endl;
            min = bbox.min();
            max = bbox.max();

            Eigen::Vector<double, DIM> size = 0.5d * bbox.diagonal();

            //find the quadrant that the next src idx belongs to

            auto tuple_idx = compute_tuple_idx(j);
            min.array() += size.array() * tuple_idx.array();
            max = min + size;

            BoundingBox<DIM> child_bbox(min, max);

            //std::cout<<"building child "<<j<<" at"<<child_bbox.min().transpose()<<" "<<child_bbox.max().transpose()<<std::endl;

            size_t end_src = src_idx;
            size_t end_target = target_idx;

            while (end_src < src_range.second && child_bbox.contains(m_srcs.col(end_src))) {
                ++end_src;
            }

            while (end_target < target_range.second && child_bbox.contains(m_targets.col(end_target))) {
                ++end_target;
            }


            //assert(src_idx>=src_range.first && end_src <=src_range.second);
            //std::cout<<"src:"<<src_idx<<end_src<<std::endl;
            const IndexRange src(src_idx, end_src);
            const IndexRange target(target_idx, end_target);
            node->setChild(j, buildOctreeNode(node, src, target, child_bbox, level + 1));

            src_idx = end_src;
            target_idx = end_target;
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

private:
    std::shared_ptr<OctreeNode > m_root;
    std::vector<std::vector<std::shared_ptr<OctreeNode> > > m_nodes;
    std::vector<unsigned int> m_numBoxes;
    unsigned int m_levels;

    unsigned int m_depth;
    size_t m_maxLeafSize;
    PointArray m_srcs;
    PointArray m_targets;
    std::vector<size_t> m_src_permutation;
    std::vector<size_t>  m_target_permutation;
    double m_diameter;

    std::function<bool(const BoundingBox<DIM>&, const BoundingBox<DIM>&) > m_isAdmissible;

};

#endif
