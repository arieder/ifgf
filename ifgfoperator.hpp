#ifndef __IFGFOPERATOR_HPP_
#define __IFGFOPERATOR_HPP_

#include <Eigen/Dense>
#include "octree.hpp"
#include "chebinterp.hpp"
#include <tbb/queuing_mutex.h>
#include <tbb/parallel_for.h>


//#define CHECK_CONNECTIVITY
//#define TWO_GRID_ONLY

#include <memory>

template<typename T, unsigned int DIM, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Matrix<double, DIM, Eigen::Dynamic, Eigen::RowMajor> PointArray;    

    IfgfOperator(const int maxLeafSize = 100, const int order=15)
    {
        m_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);
	m_base_n_elements[0]=1;
	m_base_n_elements[1]=4;

	if constexpr (DIM==3) {
	    m_base_n_elements[2]=2;
	}
	m_baseOrder=order;
    }

    ~IfgfOperator()
    {

    }

    void init(const PointArray &srcs, const PointArray targets)
    {
        m_octree->build(srcs, targets);
       
        m_octree->sanitize();

	const double hmin=m_octree->diameter()*std::pow(0.5,m_octree->levels());

	m_smin=0.1*hmin/m_octree->diameter();
	std::cout<<"hmin="<<hmin<<" vs "<<m_smin<<std::endl;

        m_numTargets = targets.cols();
        m_numSrcs = srcs.cols();

	std::cout<<"calculating interp range"<<std::endl;
	m_octree->calculateInterpolationRange([this](double H){return static_cast<Derived *>(this)->orderForBox(H, this->m_baseOrder);},[this](double H){return this->m_base_n_elements;});

    }

    Eigen::Vector<T, Eigen::Dynamic> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	//tbb::global_control control(
	//			    tbb::global_control::max_allowed_parallelism, 1);

	
        unsigned int baseOrder = m_baseOrder;       
	Eigen::Vector<size_t,DIM> n_el=m_base_n_elements;

	std::cout<<"mult"<<baseOrder<<std::endl;

	std::cout<<"permutation"<<std::endl;
        Eigen::Vector<T, Eigen::Dynamic> new_weights = Util::copy_with_permutation<T, 1> (weights, m_octree->src_permutation());
        Eigen::Vector<T, Eigen::Dynamic> result(m_numTargets);
        result.fill(0);
        unsigned int level = m_octree->levels() - 1;

	std::cout<<"boxes="<<m_octree->numBoxes(level)<<std::endl;
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM> > interpolationData(m_octree->numBoxes(level));
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM> > parentInterpolationData;

	std::cout<<"now go"<<std::endl;


        tbb::queuing_mutex resultMutex;



#ifdef CHECK_CONNECTIVITY
        Eigen::MatrixXi connectivity(m_numSrcs, m_numTargets);
        connectivity.fill(0);
#endif

        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level)),
        [&](tbb::blocked_range<size_t> r) {
	    int oldOrder = baseOrder;
            PointArray chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(baseOrder);
            PointArray transformedNodes(DIM, chebNodes.cols());
            Eigen::Vector<T, Eigen::Dynamic> tmp_result;

            for (size_t i = r.begin(); i < r.end(); i++)
                //             for(size_t i=0;i<m_octree->numBoxes(level);i++)
            {
                IndexRange srcs = m_octree->sources(level, i);
                const size_t nS = srcs.second - srcs.first;

		if(nS==0) //skip empty boxes
		    continue;

                BoundingBox bbox = m_octree->bbox(level, i);
                auto center = bbox.center();
                double H = bbox.sideLength();

                const int order = static_cast<Derived *>(this)->orderForBox(H, baseOrder);
                if (order != oldOrder) {
		    //std::cout<<"Interpolation"<<order<<std::endl;
                    chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order);
                    transformedNodes.resize(DIM, chebNodes.cols());
                    oldOrder = order;
                }

                std::vector<IndexRange> targetList = m_octree->neighborTargets(level, i);

                for (const auto &targets : targetList) {
                    const size_t nT = targets.second - targets.first;

#ifdef CHECK_CONNECTIVITY

                    for (int l = targets.first; l < targets.second; l++) {
                        for (int k = srcs.first; k < srcs.second; k++) {
                            connectivity(k, l) += 1;
                        }
                    }
#endif

                    //std::cout<<"srcs="<<srcs.first<<" "<<srcs.second<<" "<<new_weights.size()<<std::endl;
                    //std::cout<<"targets="<<targets.first<<" "<<targets.second<<" "<<std::endl;

                    tmp_result.resize(nT);
                    tmp_result.fill(0);

                    static_cast<Derived *>(this)
                    ->evaluateKernel(m_octree->sourcePoints(srcs), m_octree->targetPoints(targets), new_weights.segment(srcs.first, nS),  tmp_result);

                    {
                        tbb::queuing_mutex::scoped_lock lock(resultMutex);
                        result.segment(targets.first, nT) += tmp_result;
                    }

                }

                	       
		auto grid=m_octree->coneDomain(level,i);

		//std::cout<<"grid="<<grid.domain().min().transpose()<<std::endl;
                interpolationData[i].order = order;
		interpolationData[i].grid=grid;

		assert(chebNodes.cols()==std::pow(order,DIM));

		//std::cout<<"interp"<<std::endl;
		interpolationData[i].values.resize(grid.activeCones().size()*chebNodes.cols());		
		const size_t stride=chebNodes.cols();
		for (int memId =0;memId<interpolationData[i].grid.activeCones().size();memId++) {
		    //std::cout<<"mem"<<memId;
		    const size_t el=interpolationData[i].grid.activeCones()[memId];
		    
		    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, center, H);
		    interpolationData[i].values.segment(memId*stride,stride) =
			static_cast<const Derived *>(this)->evaluateFactoredKernel(m_octree->sourcePoints(srcs), transformedNodes, new_weights.segment(srcs.first, nS), center, H);
		}

            }
        });

        for (; level >= 2; --level) {
            std::cout << "level=" << level << std::endl;
            interpolationData.resize(m_octree->numBoxes(level));
            
            //evaluate for the cousin targets using the interpolated data
            std::cout << "step 1" << std::endl;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level)),
            [&](tbb::blocked_range<size_t> r) {
                Eigen::Vector<T, Eigen::Dynamic> tmp_result;
                for (size_t i = r.begin(); i < r.end(); i++) {

                    if(!m_octree->hasSources(level,i))
                        continue;

                    BoundingBox bbox = m_octree->bbox(level, i);
                    auto center = bbox.center();
                    double H = bbox.sideLength();
                    IndexRange srcs = m_octree->sources(level, i);

                    const std::vector<IndexRange> cousinTargets = m_octree->cousinTargets(level, i);
                    for (unsigned int l = 0; l < cousinTargets.size(); l++) {
                        const size_t nT = cousinTargets[l].second - cousinTargets[l].first;

#ifdef CHECK_CONNECTIVITY
                        for (int q = cousinTargets[l].first; q < cousinTargets[l].second; q++) {
                            for (int k = srcs.first; k < srcs.second; k++) {
                                connectivity(q, k) += 1;
                            }
                        }
#endif
                        tmp_result.resize(nT);
			
                        evaluateFromInterp(interpolationData[i], m_octree->targetPoints(cousinTargets[l]), center, H,
                                           tmp_result);

                        {
                            tbb::queuing_mutex::scoped_lock lock(resultMutex);
                            result.segment(cousinTargets[l].first, nT) += tmp_result;
                        }
                    }
                }
            });
            std::cout << "step 2" << std::endl;

            //std::cout<<"connectivity"<<std::endl<<connectivity<<std::endl;

            //Now transform the interpolation data to the parents

            parentInterpolationData.resize(m_octree->numBoxes(level - 1));
	    std::vector<tbb::queuing_mutex> interpDataMutex(m_octree->numBoxes(level - 1));
            for (int pId = 0; pId < parentInterpolationData.size(); pId++) {
		if (!m_octree->hasSources(level-1, pId)) {
                        continue;
		}

                BoundingBox bbox = m_octree->bbox(level - 1, pId);
		//std::cout<<"bbox="<<bbox.min().transpose()<<" "<<bbox.max().transpose()<<std::endl;
                auto center = bbox.center();
                double H = bbox.sideLength();

                const int order = static_cast<Derived *>(this)->orderForBox(H, baseOrder);
		auto grid= m_octree->coneDomain(level-1,pId);		
		parentInterpolationData[pId].grid = grid;
		parentInterpolationData[pId].values.resize(grid.activeCones().size()*pow(order,DIM));
                //parentInterpolationData[pId].values.resize(ChebychevInterpolation::chebnodesNdd<double, DIM>(order).cols());
                parentInterpolationData[pId].values.fill(0);
                parentInterpolationData[pId].order = order;

		//std::cout<<"Interpolation"<<order<<std::endl;

            }
	    

            std::cout << "step 3" << std::endl;
	    if(level<=2) //we dont need the interpolation info for those levels.
		continue; 
	    
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level),32),
            [&](tbb::blocked_range<size_t> r) {
                PointArray chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(baseOrder);
                PointArray transformedNodes(DIM, chebNodes.cols());
                Eigen::Vector<T, Eigen::Dynamic> tmpInterpolationData;
                int old_p_order = -1;
		int skipped=0;
                for (size_t i = r.begin(); i < r.end(); i++)
                    //            for(size_t i=0;i<m_octree->numBoxes(level);i++)
                {
                    if (!m_octree->hasSources(level, i)) {
			//skipped++;
                        continue;
                    }
                    //current node
                    BoundingBox bbox = m_octree->bbox(level, i);
                    auto center = bbox.center();
                    double H = bbox.sideLength();

                    //parent node
                    size_t parentId = m_octree->parentId(level, i);
                    BoundingBox parent_bbox = m_octree->bbox(level - 1, parentId);
                    auto parent_center = parent_bbox.center();
                    double pH = parent_bbox.sideLength();

                    const int p_order = parentInterpolationData[parentId].order;
                    if (p_order != old_p_order) {
                        chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(p_order);
                        transformedNodes.resize(DIM, chebNodes.cols());

                        old_p_order = p_order;
                    }


		    auto grid=parentInterpolationData[parentId].grid;
                    tmpInterpolationData.resize(grid.activeCones().size()*chebNodes.cols());
                    tmpInterpolationData.fill(0);
#ifdef TWO_GRID_ONLY
                    IndexRange srcs = m_octree->sources(level, i);
                    int nS = srcs.second - srcs.first;

		    
		    const size_t stride=chebNodes.cols();

		    if(!grid.isEmpty()) {
			for (int memId =0;memId<grid.activeCones().size();memId++) {
			    const size_t el=grid.activeCones()[memId];
			    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, parent_center, pH);
			    tmpInterpolationData.segment(memId*stride,stride) =
				static_cast<const Derived *>(this)->evaluateFactoredKernel(m_octree->sourcePoints(srcs), transformedNodes, new_weights.segment(srcs.first, nS), parent_center, pH);
			}
		    }

                    //tmpInterpolationData =
                    //    static_cast<const Derived *>(this)->evaluateFactoredKernel(m_octree->sourcePoints(srcs), transformedNodes, new_weights.segment(srcs.first, nS), parent_center, pH);
#else
		    if(!grid.isEmpty()) {
			const size_t stride=chebNodes.cols();
			for (int memId =0;memId<grid.activeCones().size();memId++) {
			    const size_t el=grid.activeCones()[memId];
			    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, parent_center, pH);			
			    transferInterp(interpolationData[i], transformedNodes, center, H, parent_center, pH, tmpInterpolationData.segment(memId*stride,stride));			
			}
		    }
#endif

                    //Free the data we no longer use
                    interpolationData[i].values.resize(0);
                    {
                        assert(parentInterpolationData[parentId].values.size() == tmpInterpolationData.size());
                        tbb::queuing_mutex::scoped_lock lock(interpDataMutex[parentId]);
                        parentInterpolationData[parentId].values.matrix() += tmpInterpolationData;
                    }
                }
		//std::cout<<"skipped "<<skipped<<" boxes out of"<<r.end()-r.begin()<<std::endl;
            });

            std::swap(interpolationData, parentInterpolationData);
            parentInterpolationData.resize(0);

        }
#ifdef CHECK_CONNECTIVITY
        assert((connectivity.array() - 1).matrix().norm() < 1e-10);
#endif

        return Util::copy_with_inverse_permutation<T, 1>(result, m_octree->target_permutation());
    }

    inline void transferInterp(const ChebychevInterpolation::InterpolationData<T,DIM>& data, const Eigen::Ref<const PointArray> &targets,
			       const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                               const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                               Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > result) const
    {
        //transform to the child interpolation domain
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

	result.fill(0);
	const size_t stride=std::pow(data.order,DIM);
	//std::cout<<"stride"<<stride<<std::endl;
	size_t idx=0;
	while (idx<transformed.cols())
	{
	    size_t nb=1;
	    const size_t el=data.grid.elementForPoint(transformed.col(idx));
	    const size_t memId=data.grid.memId(el);
	    
	    transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	    //look if any of the following points are also in this element. that way we can process them together
	    while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
		transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));		
		nb++;
	    }
	    ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array().middleCols(idx,nb), data.values.segment(memId*stride,stride), result.segment(idx,nb), data.order);
	    idx+=nb;
	}
	
        for (unsigned int j = 0; j < targets.cols(); j++) {
            const T tf = static_cast<const Derived *>(this)->transfer_factor(targets.col(j), xc, H, p_xc, pH);
            result[j] = tf*result[j];
        }

    }

    inline void evaluateFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM>& data, const Eigen::Ref<const PointArray> &targets,
				   const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                                   Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > result) const
    {
	//std::cout<<"efromInt"<<data.order<<std::endl;
	//sort the points into the corresponding cones
	PointArray transformed(DIM, targets.cols());
	transformCartToInterp(targets, transformed, xc, H);
	const size_t stride=std::pow(data.order,DIM);
	//std::cout<<"stride"<<stride<<std::endl;
	size_t idx=0;
	while (idx<transformed.cols())
	{
	    //std::cout<<"idx"<<idx<<std::endl;
	    size_t nb=1;	    
	    const size_t el=data.grid.elementForPoint(transformed.col(idx));	    
	    const size_t memId=data.grid.memId(el);
	   

	    //std::cout<<"el="<<el<<" "<<transformed.col(idx)<<std::endl;
	    transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	    //look if any of the following points are also in this elemnt. that way we can process them together
	    while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
		transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));
		nb++;
	    }
	    //std::cout<<"bla"<<data.values.segment(memId*stride,stride)<<std::endl;
	    ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array().middleCols(idx,nb), data.values.segment(memId*stride,stride), result.segment(idx,nb), data.order);
	    idx+=nb;
	}
	
		
        for (unsigned int j = 0; j < targets.cols(); j++) {
            const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j) - xc);
            result[j] *= cf;
        }
	//std::cout<<"done"<<data.order<<std::endl;
    }


    void transformCartToInterp(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {
        for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = Util::cartToInterp<DIM>(nodes.col(i), xc, H);
        }
    }

    void transformInterpToCart(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {

        for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = Util::interpToCart<DIM>(nodes.col(i), xc, H);
        }
    }

private:
    std::unique_ptr<Octree<T, DIM> > m_octree;
    unsigned int m_numTargets;
    unsigned int m_numSrcs;
    double m_smin;
    Eigen::Vector<size_t, DIM> m_base_n_elements;
    size_t m_baseOrder;
};

#endif