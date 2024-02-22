#ifndef _OCTREE_HPP_
#define _OCTREE_HPP_

#include <Eigen/Dense>
#include <memory>
#include <map>
#include <vector>
#include <execution>

#include "util.hpp"
#include "boundingbox.hpp"
#include "zorder_less.hpp"
#include "chebinterp.hpp"

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
        size_t m_id;
        std::shared_ptr<OctreeNode> m_children[N_Children];
        std::vector<std::shared_ptr<const OctreeNode> > m_neighbors;

        IndexRange m_srcRange;
        IndexRange m_targetRange;

	ChebychevInterpolation::ConeDomain<DIM> m_coneDomain;
	//std::map<ConeIndex,Cone> m_relevant_cones;

        BoundingBox<DIM> m_bbox;

        bool m_dirty;
        bool m_isLeaf;
        unsigned int m_level;
    public:

        OctreeNode(std::shared_ptr<OctreeNode> parent, unsigned int level) :
            m_parent(parent),
            m_dirty(false),
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

        size_t id() const
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

        void addNeighbor(const std::shared_ptr<const OctreeNode>   &nb)
        {
            m_neighbors.push_back(nb);
        }

        std::vector<std::shared_ptr<const OctreeNode> > neighbors() const
        {
            return m_neighbors;
        };

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
            m_children[idx] = node;
            m_isLeaf = false;
            m_dirty = true;
        }

	void setConeDomain(const ChebychevInterpolation::ConeDomain<DIM>& domain)
	{
	    m_coneDomain=domain;
	}

	const ChebychevInterpolation::ConeDomain<DIM>& coneDomain() const
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
            //assert(!m_dirty);
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

        unsigned int level() const
        {
            return m_level;
        }

        const std::shared_ptr<const OctreeNode> parent() const
        {
            return m_parent;
        }

        void print(std::string prefix = "") const
        {
            std::cout << prefix;
            std::cout << "----";

            std::cout << m_id << " ";
            std::cout << "(" << m_targetRange.first << " " << m_targetRange.second << ")"; //<<std::endl;
            std::cout << m_bbox.min().transpose() << "   " << m_bbox.max().transpose() << std::endl;

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
                        std::cout << prefix + "   ----x" << std::endl;
                    }
                }
            }
        }

    };

    typedef std::pair<size_t, size_t> BoxIndex; //level + index in level

    Octree(int maxLeafSize):
        m_maxLeafSize(maxLeafSize)
    {

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
	//We are now left with a sparse octree. Our algorithms require it to be balanced, so we refine some more up to the finished depth
	//fillupOctree(m_root);
	
        std::cout << "building the nodes up to level"<<m_depth << std::endl;
        

        buildNeighborList(m_root);

        //m_root->print();
    }

    void buildNeighborList(std::shared_ptr<OctreeNode>  node)
    {
        //go through the parents neighbours childen and check if they touch this node
        const std::shared_ptr<const OctreeNode >parent = node->parent();
        if (parent == 0) {
            node->addNeighbor(node); //we are always in our own neighborhood
        } else {

            for (const auto p_nb : parent->neighbors()) {
                for (int j = 0; j < N_Children; j++) {
                    const std::shared_ptr<const OctreeNode > neighbor = p_nb->child(j);
                    if (neighbor && node->boundingBox().squaredExteriorDistance(neighbor->boundingBox()) <= std::numeric_limits<double>::epsilon()) {
                        node->addNeighbor(neighbor);
                    }
                }
            }
        }

        //std::cout<<"got "<<node->neighbors().size()<<" neighbors"<<std::endl;

        if (node->isLeaf()) {
            return;
        }

        //now compute the neighbors of the children
        for (int j = 0; j < N_Children; j++) {
            buildNeighborList(node->child(j));
        }

    }

    void calculateInterpolationRange(  std::function<size_t(double)> order_for_H,std::function<Eigen::Vector<size_t,DIM>(double)> N_for_H)
    {
	//No interpolation at the two highest levels 
	for (size_t level=2;level<levels();level++) {
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
		const std::vector<IndexRange> cT=cousinTargets(node);

		for(const IndexRange& iR : cT)
		{				    
		    for(int i=iR.first;i<iR.second;i++) {
			const auto s=Util::cartToInterp<DIM>(m_targets.col(i),xc,H);
			box.extend(s); //make sure the target is in the interpolation domain
		    }
		}

		//now also add all the parents targets
		const BoundingBox pBox=node->parent()->interpolationRange();
		const Point pxc=node->parent()->boundingBox().center();
		const double pH=node->parent()->boundingBox().sideLength();


		if(!pBox.isNull())
		{

		    //transform the parents interpolation range to the physical coordinates
		    auto cMin=Util::interpToCart<DIM>(pBox.min(),pxc,pH);
		    auto cMax=Util::interpToCart<DIM>(pBox.max(),pxc,pH);

	       

		    //pull those physical coordinates back to the interpolation-coordinates of node
		    box.extend(Util::cartToInterp<DIM>(cMin,xc,H));	
		    box.extend(Util::cartToInterp<DIM>(cMax,xc,H));

		    const ChebychevInterpolation::ConeDomain<DIM>& p_grid=node->parent()->coneDomain();
		    auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH));
		    for(size_t el : p_grid.activeCones() ) {			
			for (size_t i=0;i<chebNodes.cols();i++) {
			    auto pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)),pxc,pH);
			    box.extend(Util::cartToInterp<DIM>(pnt,xc,H));
			}
		    }
		}


		//std::cout<<"box="<<box<<std::endl;
		ChebychevInterpolation::ConeDomain<DIM> domain(N_for_H(H),box);

		//now we need to do the whole thing again to figure out which cones are active...
		std::vector<bool> is_cone_active(domain.n_elements());
		std::fill(is_cone_active.begin(),is_cone_active.end(),false);
		
		for(const IndexRange& iR : cT)
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
		    const ChebychevInterpolation::ConeDomain<DIM>& p_grid=node->parent()->coneDomain();		    
		    auto chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order_for_H(pH));

		    for(size_t el : p_grid.activeCones() ) {			
		    	for (size_t i=0;i<chebNodes.cols();i++) {
			    auto cart_pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)),pxc,pH);
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

		domain.setActiveCones(active_cones);
		domain.setConeMap(cone_map);

		//std::cout<<"level"<<node->level()<<"range:"<<box.min().transpose()<<" "<<box.max().transpose()<<std::endl;
		node->setConeDomain(domain);
		//node->setInterpolationRange(box);
	    }});
	}
    
    }


	
    
    inline double diameter () const
    {
	return m_diameter;
    }

    unsigned int levels() const
    {
        return m_levels;
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
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];

        return node->srcRange();
    }

    const IndexRange siblingSources(unsigned int level, size_t i) const
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
	return cousinTargets(node);
    }

    const std::vector<IndexRange> cousinTargets(std::shared_ptr<OctreeNode> node) const
    {        
        std::vector<IndexRange> cousins;

        //add some of the children of the parents neighbours
        for (auto p_nb : node->parent()->neighbors()) {
            //add all the (possible) cousins which have positive distance to the current node (i.e., not neighbors)
            for (int l = 0; l < N_Children; l++) {
                const std::shared_ptr<const OctreeNode > cousin = p_nb->child(l);
                if (cousin && node->boundingBox().squaredExteriorDistance(cousin->boundingBox()) > std::numeric_limits<double>::epsilon()) {
                    //std::cout<<"cousin "<<cousin->id()<<std::endl;
                    cousins.push_back(cousin->targetRange());
                }
            }

        }

        return cousins;

    }

    const BoundingBox<DIM> bbox(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->boundingBox();
    }


    const BoundingBox<DIM> interpolationRange(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->interpolationRange();
    }

    const ChebychevInterpolation::ConeDomain<DIM> coneDomain(unsigned int level, size_t i) const
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

    void sanitize()
    {
        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_srcs.cols());
            indices.fill(0);
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->srcRange().first;
                size_t b = m_nodes[level][i]->srcRange().second;

                for (int l = a; l < b; l++) {
                    indices[l] += 1;
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

        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_targets.cols());
            indices.fill(0);
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->targetRange().first;
                size_t b = m_nodes[level][i]->targetRange().second;

                for (int l = a; l < b; l++) {
                    indices[l] += 1;
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
        if (level >= m_levels) {
            m_levels++;
            m_nodes.push_back(std::vector<std::shared_ptr<OctreeNode> >());
            m_numBoxes.push_back(0);
        }

	auto node = std::make_shared<OctreeNode >(parent, level);
	
	node->setSrcRange(src_range);
	node->setTargetRange(target_range);
	    
	node->setBoundingBox(bbox);

	m_numBoxes[level] += 1;
	node->setId(m_nodes[level].size());
	m_nodes[level].push_back(node);
	    
	if (level ==  m_depth) {
	    return node;
	}

	//std::cout<<"building node"<<src_range.first<<" to "<<src_range.second<<" and "<<target_range.first<<" to "<<target_range.second<<std::endl;
	if( m_depth== -1 &&
	    src_range.second-src_range.first<=m_maxLeafSize &&
	    target_range.second-target_range.first<=m_maxLeafSize ) {
	    //std::cout<<"leaf"<<src_range.second-src_range.first<<std::endl;
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

    void fillupOctree(std::shared_ptr<OctreeNode > node)
    {
	if (node->level() == m_depth) {
	    return;
	}

	//if we are not in a leaf, just recurse down
	if(!node->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		fillupOctree(node->child(j));
	    }
	}
	else //switch to building more of the octree
	{
	    IndexRange src_range=node->srcRange();
	    IndexRange target_range=node->targetRange();
		
	    size_t src_idx = src_range.first;
	    size_t target_idx = target_range.first;

	    BoundingBox<DIM> bbox=node->boundingBox();
	    size_t level=node->level();

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
	}


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
};

#endif
