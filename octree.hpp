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

typedef std::pair<size_t, size_t> IndexRange;

#define MAX_DEPTH 6

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

        std::cout << "building the nodes" << std::endl;
        m_levels = 0;
        m_root = buildOctreeNode(0, std::make_pair(0, m_srcs.cols()), std::make_pair(0, m_targets.cols()), bbox);

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

    const size_t parentId(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->parent()->id();
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

        if (level == MAX_DEPTH) {
            return node;
        }

        //std::cout<<"building node"<<src_range.first<<" to "<<src_range.second<<" and "<<target_range.first<<" to "<<target_range.second<<std::endl;
        /*if( src_range.second-src_range.first<=m_maxLeafSize &&
            target_range.second-target_range.first<=m_maxLeafSize ) {
            //std::cout<<"leaf"<<std::endl;
            return node;
            }*/

        size_t src_idx = src_range.first;
        size_t target_idx = target_range.first;

        for (int j = 0; j < N_Children; j++) {
            Point min;
            Point max;

            //std::cout<<"finding bbox from parent"<<std::endl;
            min = bbox.min();
            max = bbox.max();

            Eigen::Vector<double, DIM> size = 0.5 * (max - min);

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

    size_t m_maxLeafSize;
    PointArray m_srcs;
    PointArray m_targets;
    std::vector<size_t> m_src_permutation;
    std::vector<size_t>  m_target_permutation;
};

#endif
