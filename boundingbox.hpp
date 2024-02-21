#ifndef __BBOX_HPP_
#define __BBOX_HPP_

#include <Eigen/Dense>

#include <Eigen/Geometry>

template <size_t DIM>
class BoundingBox : public Eigen::AlignedBox<double, DIM>
{
public:
    BoundingBox()
    {

    }

    BoundingBox(Eigen::Vector<double, DIM> min, Eigen::Vector<double, DIM> max):
        Eigen::AlignedBox<double, DIM>(min, max)
    {

    }

    ~BoundingBox()
    {

    }

    inline Eigen::Vector<double, DIM> center() const
    {
        return  (this->max() + this->min())/2.0;
    }

    inline double sideLength() const
    {
        auto diag = this->diagonal();
        double m = 0;
        for (unsigned int i = 0; i < DIM; i++) {
            m = std::max(m, diag[i]);
        }
        return m;
    }

    
    /*

    inline void absorb(const BoundingBox& other)
    {
    m_min=m_min.min(other.min());
    m_max=m_max.max(other.max());
    }

    Eigen::Vector<double,DIM> min() const
    {
    return m_min;
    }

    Eigen::Vector<double,DIM> max() const
    {
    return m_max;
    }


    inline double dist(const BoundingBox& other) const
    {
    //TODO

    return 0;
    }

    inline double dist(const Eigen::Vector<double,DIM>& x) const
    {
    //TODO
    return 0;
    }


    bool contains(const Eigen::Vector<double,DIM>& x)  const
    {
    return x>=m_min && x<=m_max;
    }




    private:
    Eigen::Vector<double,DIM> m_min;
    Eigen::Vector<double,DIM> m_max;
    */
};

#endif
