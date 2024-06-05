#ifndef WEIGHTEDOBJECT_H
#define WEIGHTEDOBJECT_H

namespace CluE
{

/**
 * @brief Abstract base class for weighted objects
 */
class WeightedObject
{
public:
	virtual size_t getWeight() const = 0;
	virtual void setWeight(size_t w) = 0;
};

}

#endif
