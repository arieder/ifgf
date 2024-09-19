#ifndef _CONE_DOMAIN_HPP_
#define _CONE_DOMAIN_HPP_

#include <Eigen/Dense>
#include <memory>
#include <map>
#include <iostream>

#include "boundingbox.hpp"
#include "util.hpp"

class ConeRef
{
public:
    ConeRef(size_t level, size_t id,size_t memId, size_t boxId):
	m_level(level),
	m_id(id),
	m_memId(memId),
	m_boxId(boxId)
    {

    }

    size_t level() const {
	return m_level;
    }


    //index in the full NxNxN grid
    size_t id() const
    {
	return m_id;
    }


    //index in memory (i.e. when skipping al non-active cones
    size_t memId() const
    {
	return m_memId;
    }

    


    size_t boxId() const
    {
	return m_boxId;
    }

private:
    size_t m_level;
    size_t m_id;
    size_t m_boxId;
    size_t m_memId;

};
    


template<size_t DIM>
class ConeDomain
{
    typedef Eigen::Matrix<double, DIM, Eigen::Dynamic> PointArray;
    
public:
    ConeDomain()
    {

    }

    ConeDomain(Eigen::Vector<size_t, DIM> numEls, const BoundingBox<DIM>& domain) :
	m_numEls(numEls),
	m_domain(domain)
    {
    }

    inline void setNElements(Eigen::Vector<size_t, DIM> numEls) {
	m_numEls=numEls;	
    }

    inline constexpr size_t n_elements() const
    {
	size_t n=1;
	for(size_t i=0;i<DIM;i++)
	    n*=m_numEls[i];
	return n;
    }

    inline constexpr size_t n_elements(size_t d) const {
	return m_numEls[d];
    }
    

    inline const std::vector<size_t>& activeCones() const
    {
	return m_activeCones;
    }

    inline void setActiveCones( std::vector<size_t>& cones)
    {
	m_activeCones=cones;
        precomputeRegions();
    }

    void precomputeRegions()
    {
        m_regions.resize(m_activeCones.size());
        size_t idx=0;
        for( size_t j: m_activeCones )
        {
            auto j0=j;

            assert(j<n_elements());
            Eigen::Vector<double, DIM> min,max;
            Eigen::Vector<double, DIM> h=m_domain.diagonal();
            for(int i=0;i<DIM;i++) {
                const size_t idx=j % m_numEls[i];
                j=j / m_numEls[i];

	    
	    
                min[i]=m_domain.min()[i]+(idx*(h[i]/((double) m_numEls[i])));
                max[i]=(min[i]+(h[i]/((double) m_numEls[i])));
            }

            

            m_regions[idx]= BoundingBox<DIM>(min,max);
            idx++;
        }
    }


    inline void setConeMap( std::unordered_map<size_t,size_t>& cone_map)
    {
	m_coneMap=cone_map;
    }


    inline size_t memId(size_t el) const
    {
	assert(isActive(el));
	return m_coneMap.at(el);
    }


    inline bool isActive(size_t el) const
    {
	return m_coneMap.count(el)>0;
    }
    
    inline BoundingBox<DIM> domain() const {
	return m_domain;
    }

    //transforms from (-1,1) to K_el
    Eigen::Matrix<double,DIM,Eigen::Dynamic> transform(size_t el,const Eigen::Ref<const PointArray >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());
	const auto a=0.5*(bbox.max()-bbox.min()).array();
	const auto b=0.5*(bbox.max()+bbox.min()).array();

	for(int i=0;i<pnts.cols();i++) {
	    tmp.col(i)=(pnts.array().col(i)*a)+b;
	}

	/*	for(size_t j=0;j<pnts.cols();j++)
	  {

	    tmp.col(j)=pnts.col(j).array()*a+b;

	    assert(bbox.exteriorDistance(tmp.col(j))<1e-14);
	    }*/
	
	
	return tmp;
	    
    }



    //transforms from K_el to (-1,1)
    inline Eigen::Matrix<double,DIM,Eigen::Dynamic> transformBackwards(size_t el,const Eigen::Ref<const  PointArray >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	const auto a=0.5*(bbox.max()-bbox.min()).array();
	const auto b=0.5*(bbox.max()+bbox.min()).array();

	tmp.array()=(pnts.array().colwise()-b).colwise()/a;
	
	/*for(size_t j=0;j<pnts.cols();j++)
	{	    
	    assert(bbox.squaredExteriorDistance(pnts.col(j))<1e-12);
	    
	    tmp.col(j)=(pnts.col(j).array()-b)/a;

	    
	    assert(-1-1e-8<=tmp.col(j)[0] && tmp.col(j)[0]<=1+1e-8);
	    //assert(-1-1e-8<=tmp.col(j)[1] && tmp.col(j)[1]<=1+1e-8);
	    }*/
	
	return tmp;	    
    }

    inline Eigen::Vector<size_t,DIM> indicesFromId(size_t j) const {
	Eigen::Vector<size_t,DIM> indices;	
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];

	    
	    indices[i]=idx;
	}

	return indices;
    }


    inline BoundingBox<DIM> region(size_t j) const
    {
        return m_regions.at(memId(j));
    }

    inline size_t elementForPoint(const Eigen::Ref<const Eigen::Vector<double,DIM> > & pnt) const
    {
	size_t idx=0;
	int stride=1;
        /*if(m_domain.squaredExteriorDistance(pnt)>1e-8) {
	std::cout<<"dom:"<<m_domain<<std::endl;
	//std::cout<<m_numEls<<std::endl;
	std::cout<<"pn"<<pnt.transpose()<<std::endl;
        }*/
	assert(m_domain.squaredExteriorDistance(pnt)<1e-8);
	for(int j=0;j<DIM;j++) {	    
	    const double q=(pnt[j]-m_domain.min()[j])*m_numEls[j]/m_domain.diagonal()[j];
	    
	    //std::cout<<"floor"<<std::floor(q)<<std::endl;
	    const size_t ij=static_cast<int>( std::floor(std::clamp(q,0.0, m_numEls[j]-1.0)));
	    //std::cout<<q<<"ij "<<ij<<std::endl;
	    idx+=ij*stride;
	    stride*=m_numEls[j];
	}

	assert(idx<n_elements());
	return idx;

    }

    bool isEmpty() const
    {
	return m_domain.isEmpty();
    }


    inline Eigen::Matrix<double,DIM,Eigen::Dynamic> rotated_points(size_t el,const Eigen::Ref<const  PointArray >& pnts, Eigen::Vector3d direction,bool backward) const
    {
	Eigen::Vector3d xc=Eigen::Vector3d::Zero();
	const double H=1;
       
	auto targets=Util::interpToCart<DIM>(transform(el,pnts).array(), xc, H);

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
	PointArray transformed(3,targets.cols());
	Util::cartToInterp2<DIM>(targets.array(),  xc, H,transformed);

	return transformed;
    }


    inline Eigen::Matrix<double,DIM,Eigen::Dynamic> translated_points(size_t el,const Eigen::Ref<const  PointArray >& pnts, Eigen::Vector3d xc, double H, Eigen::Vector3d pxc, double pH) const
    {	
	auto targets=Util::interpToCart<DIM>(transform(el,pnts).array(), xc-Eigen::Vector3d(0,0,(pxc-xc).norm()), pH);
	    
	//transform to the child interpolation domain
	PointArray transformed(3,targets.cols());
	Util::cartToInterp2<DIM>(targets.array(), xc, H,transformed);
	return transformed;

    }

private:
    BoundingBox<DIM> m_domain;
    Eigen::Vector<size_t, DIM> m_numEls;
    std::vector<size_t> m_activeCones;
    std::unordered_map<size_t,size_t> m_coneMap;
    std::vector<BoundingBox<DIM> > m_regions;
};




#endif
