#ifndef __IFGFOPERATOR_HPP_
#define __IFGFOPERATOR_HPP_

#include <Eigen/Dense>
#include "octree.hpp"
#include "chebinterp.hpp"
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>

//#define CHECK_CONNECTIVITY
//#define TWO_GRID_ONLY

#include <memory>

template<typename T, unsigned int DIM, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Matrix<double, DIM, Eigen::Dynamic, Eigen::RowMajor> PointArray;

    IfgfOperator(const int maxLeafSize = 100)
    {
        m_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);

    }

    ~IfgfOperator()
    {

    }

    void init(const PointArray &srcs, const PointArray targets)
    {
        m_octree->build(srcs, targets);

        m_octree->sanitize();

        m_numTargets = targets.cols();
        m_numSrcs = srcs.cols();
    }

    Eigen::Vector<T, Eigen::Dynamic> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
        unsigned int baseOrder = 30;

        Eigen::Vector<T, Eigen::Dynamic> new_weights = Util::copy_with_permutation<T, 1> (weights, m_octree->src_permutation());
        Eigen::Vector<T, Eigen::Dynamic> result(m_numTargets);
        result.fill(0);
        unsigned int level = m_octree->levels() - 1;

        std::vector<ChebychevInterpolation::InterpolationData<T> > interpolationData(m_octree->numBoxes(level));
        std::vector<ChebychevInterpolation::InterpolationData<T> > parentInterpolationData;

        tbb::spin_mutex resultMutex;
        tbb::spin_mutex interpDataMutex;

        int oldOrder = baseOrder;
#ifdef CHECK_CONNECTIVITY
        Eigen::MatrixXi connectivity(m_numSrcs, m_numTargets);
        connectivity.fill(0);
#endif
        PointArray transformedNodes(DIM, baseOrder);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level)),
        [&](tbb::blocked_range<size_t> r) {
            PointArray chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(baseOrder);
            PointArray transformedNodes(DIM, chebNodes.cols());
            Eigen::Vector<T, Eigen::Dynamic> tmp_result;

            for (size_t i = r.begin(); i < r.end(); i++)
                //             for(size_t i=0;i<m_octree->numBoxes(level);i++)
            {
                IndexRange srcs = m_octree->sources(level, i);
                const size_t nS = srcs.second - srcs.first;

                BoundingBox bbox = m_octree->bbox(level, i);
                auto center = bbox.center();
                double H = bbox.sideLength();

                const int order = static_cast<Derived *>(this)->orderForBox(H, baseOrder);
                if (order != oldOrder) {
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
                        tbb::spin_mutex::scoped_lock lock(resultMutex);
                        result.segment(targets.first, nT) += tmp_result;
                    }

                }

                transformInterpToCart(chebNodes, transformedNodes, center, H);

                interpolationData[i].order = order;
                interpolationData[i].values =
                    static_cast<const Derived *>(this)->evaluateFactoredKernel(m_octree->sourcePoints(srcs), transformedNodes, new_weights.segment(srcs.first, nS), center, H);

            }
        });

        for (; level >= 2; --level) {
            std::cout << "level=" << level << std::endl;
            interpolationData.resize(m_octree->numBoxes(level));

            assert(interpolationData.size() == m_octree->numBoxes(level));
            //evaluate for the cousin targets using the interpolated data
            std::cout << "step 1" << std::endl;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level)),
            [&](tbb::blocked_range<size_t> r) {
                PointArray chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(baseOrder);
                PointArray transformedNodes(DIM, chebNodes.cols());
                Eigen::Vector<T, Eigen::Dynamic> tmp_result;
                for (size_t i = r.begin(); i < r.end(); i++) {

                    //if(!m_octree->hasSources(level,i))
                    //    continue;

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

                        evaluateFromInterp(interpolationData[i].values, m_octree->targetPoints(cousinTargets[l]),
                                           interpolationData[i].order, center, H,
                                           tmp_result);

                        {
                            tbb::spin_mutex::scoped_lock lock(resultMutex);
                            result.segment(cousinTargets[l].first, nT) += tmp_result;
                        }
                    }
                }
            });
            std::cout << "step 2" << std::endl;

            //std::cout<<"connectivity"<<std::endl<<connectivity<<std::endl;

            //Now transform the interpolation data to the parents

            parentInterpolationData.resize(m_octree->numBoxes(level - 1));
            for (int pId = 0; pId < parentInterpolationData.size(); pId++) {
                BoundingBox bbox = m_octree->bbox(level - 1, pId);
                auto center = bbox.center();
                double H = bbox.sideLength();

                const int order = static_cast<Derived *>(this)->orderForBox(H, baseOrder);

                parentInterpolationData[pId].values.resize(ChebychevInterpolation::chebnodesNdd<double, DIM>(order).cols());
                parentInterpolationData[pId].values.fill(0);
                parentInterpolationData[pId].order = order;

            }

            std::cout << "step 3" << std::endl;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m_octree->numBoxes(level)),
            [&](tbb::blocked_range<size_t> r) {
                PointArray chebNodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(baseOrder);
                PointArray transformedNodes(DIM, chebNodes.cols());
                Eigen::Vector<T, Eigen::Dynamic> tmpInterpolationData;
                int old_p_order = -1;
                for (size_t i = r.begin(); i < r.end(); i++)
                    //            for(size_t i=0;i<m_octree->numBoxes(level);i++)
                {
                    if (!m_octree->hasSources(level, i)) {
                        continue;
                    }
                    //current node
                    BoundingBox bbox = m_octree->bbox(level, i);
                    auto center = 0.5 * (bbox.max() + bbox.min());
                    double H = ((bbox.max() - bbox.min()))[0];

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

                    transformInterpToCart(chebNodes, transformedNodes, parent_center, pH); //points at which we need to evaluate the interpolation info

                    tmpInterpolationData.resize(chebNodes.cols());
                    tmpInterpolationData.fill(0);
#ifdef TWO_GRID_ONLY
                    IndexRange srcs = m_octree->sources(level, i);
                    int nS = srcs.second - srcs.first;

                    transformInterpToCart(chebNodes, transformedNodes, parent_center, pH);

                    tmpInterpolationData =
                        static_cast<const Derived *>(this)->evaluateFactoredKernel(m_octree->sourcePoints(srcs), transformedNodes, new_weights.segment(srcs.first, nS), parent_center, pH);
#else

                    transferInterp(interpolationData[i].values, transformedNodes, interpolationData[i].order, center, H, parent_center, pH, tmpInterpolationData);
#endif

                    //Free the data we no longer use
                    interpolationData[i].values.resize(0);
                    {
                        assert(parentInterpolationData[parentId].values.size() == tmpInterpolationData.size());
                        tbb::spin_mutex::scoped_lock lock(interpDataMutex);
                        parentInterpolationData[parentId].values.matrix() += tmpInterpolationData;

                    }
                }
            });

            std::swap(interpolationData, parentInterpolationData);
            parentInterpolationData.resize(0);

        }
#ifdef CHECK_CONNECTIVITY
        assert((connectivity.array() - 1).matrix().norm() < 1e-10);
#endif

        return Util::copy_with_inverse_permutation<T, 1>(result, m_octree->target_permutation());
    }

    inline void transferInterp(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &data, const Eigen::Ref<const PointArray> &targets,
                               unsigned int interp_order, const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                               const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                               Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > result) const
    {
        //transform to the child interpolation domain
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

        ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array(), data.array(), result, interp_order);
        for (unsigned int j = 0; j < targets.cols(); j++) {
            const T tf = static_cast<const Derived *>(this)->transfer_factor(targets.col(j), xc, H, p_xc, pH);
            result[j] *= tf;
        }

    }

    inline void evaluateFromInterp(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &data, const Eigen::Ref<const PointArray> &targets,
                                   unsigned int interp_order, const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                                   Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, 1> > result) const
    {
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

        // //TMP TESTING CODE
        // PointArray tmp2(DIM,targets.cols());
        // transformInterpToCart(transformed,tmp2,xc,H);
        // assert((tmp2-targets).norm()<1e-14);

        ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array(), data.array(), result, interp_order);

        for (unsigned int j = 0; j < targets.cols(); j++) {
            const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j) - xc);
            result[j] *= cf;
        }
    }

    inline Eigen::Vector<double, DIM> interpToCart(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H) const
    {
        //p is in (-1,1) we need to sample s in (1/3,sqrt(3)/3) add a corresponding m
        const double s0 = 1e-12;
        const double s1 = sqrt(DIM) / (double) DIM;

        const double s = 0.5 * ((s1 - s0) * p[0] + (s1 + s0));

        //std::cout<<"s="<<s0<<" "<<s<<" "<<s1<<" "<<p[0]<<std::endl;
        assert(s0-1e-8 <= s && s <= s1+1e-8);
        const double theta = M_PI * p[1];

        Eigen::Vector<double, DIM> res = xc;

        const double r = H / s;
        if constexpr(DIM==2) {
            res[0] += r * cos(theta);
            res[1] += r * sin(theta);
        }else {
            assert(DIM==3);
            const double phi=(M_PI/2.0)*p[2];
            res[0] += r * cos(phi)*sin(theta);
            res[1] += r * sin(phi)*sin(theta);
            res[1] += r* cos(theta);
        }

        return  res;
    }

    inline Eigen::Vector<double, DIM> cartToInterp(Eigen::Vector<double, DIM> p, const Eigen::Vector<double, DIM> &xc, double H) const
    {
        auto xp = p - xc;


        if constexpr (DIM==2) {
             const double r = xp.norm();

            const double phi = std::atan2(xp[1], xp[0]);

            const double s = H / r;

            const double s0 = 1e-12; //1e-8;
            const double s1 = sqrt(DIM) / (double) DIM;

            const double q = (s - (s1 + s0) / 2.0) / ((s1 - s0) / 2);

            Eigen::Vector<double, DIM> res;
            res[0] = q;
            res[1] = phi / (M_PI);

            assert(-1.0001 <= res[0] && res[0] <= 1.0001);
            assert(-1 <= res[1] && res[1] <= 1);

            return  res;

        }else{
            assert(DIM==3);
            const double phi = std::atan2(xp[1], xp[0]);
            const double a=(xp[0]*xp[0]+xp[1]*xp[1]);
            const double theta= std::atan2(sqrt(a),xp[2]);
            const double r= sqrt(a+xp[2]*xp[2]);

            const double s=H/r;
            const double s0=1e-12;
            const double s1 = sqrt(DIM) / (double) DIM;

            const double q = (s - (s1 + s0) / 2.0) / ((s1 - s0) / 2);

            Eigen::Vector<double, DIM> res;
            res[0] = q;
            res[1] = phi / (M_PI);
            res[2]= phi/ (2*M_PI);

            return  res;

        }

    }

    void transformCartToInterp(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {
        for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = cartToInterp(nodes.col(i), xc, H);
        }
    }

    void transformInterpToCart(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {

        for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = interpToCart(nodes.col(i), xc, H);
        }
    }

private:

    std::unique_ptr<Octree<T, DIM> > m_octree;
    unsigned int m_numTargets;
    unsigned int m_numSrcs;

};

#endif
