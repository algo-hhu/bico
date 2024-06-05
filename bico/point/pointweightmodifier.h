#ifndef POINTWEIGHTMODIFIER_H
#define POINTWEIGHTMODIFIER_H

#include "../base/weightmodifier.h"
#include "../point/point.h"

namespace CluE
{

/**
 * @brief Modifies the weight of a Point.
 *
 * @ingroup pointrelated_classes
 */
class PointWeightModifier : public WeightModifier<Point>
{
public:
	virtual PointWeightModifier* clone() const;

	virtual size_t getWeight(Point&);
	virtual void setWeight(Point&, size_t);
};

inline
PointWeightModifier* PointWeightModifier::clone() const
{
	return new PointWeightModifier(*this);
}

inline
size_t PointWeightModifier::getWeight(Point& p)
{
    return p.getWeight();
}

inline
void PointWeightModifier::setWeight(Point& p, size_t w)
{
    p.setWeight(w);
}

}

#endif
