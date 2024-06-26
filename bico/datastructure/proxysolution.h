#ifndef PROXYSOLUTION_H
#define PROXYSOLUTION_H

#include "../base/solutionprovider.h"
#include "../base/proxyprovider.h"

#include <vector>

namespace CluE
{

/**
 * @brief Data structure for proxies.
 *
 * This struct is for use in algorithms computing proxies.
 *
 * @ingroup data_structures
 */
template<typename T> struct ProxySolution : public SolutionProvider, public ProxyProvider<T>
{
public:

	ProxySolution();

	virtual ~ProxySolution()
	{
	}

	virtual double computationtime() const;
	virtual size_t number_of_solutions() const;
	virtual size_t size_of_solution(unsigned int) const;

	virtual T proxy(unsigned int n, unsigned int c) const;
	virtual std::vector<T> proxies(unsigned int n) const;

	double seconds;
	std::vector<std::vector<T>> proxysets;
};

template<typename T> ProxySolution<T>::ProxySolution() : seconds()
{
}

template<typename T> double ProxySolution<T>::computationtime() const
{
	return seconds;
}

template<typename T> size_t ProxySolution<T>::number_of_solutions() const
{
	return this->proxysets.size();
}

template<typename T> size_t ProxySolution<T>::size_of_solution(unsigned int i) const
{
	if (i<this->proxysets.size())
		return this->proxysets[i].size();
	return 0;
}

template<typename T> T ProxySolution<T>::proxy(unsigned int n, unsigned int c) const
{
	if (n<this->proxysets.size())
		if (c<this->proxysets[n].size())
				return this->proxysets[n][c];

	std::cerr << "ProxySolution<T>::proxy(" << n << "," << c << "): requested proxy not available" << std::endl;
	throw "ILLEGAL STATE";
}

template<typename T> std::vector<T> ProxySolution<T>::proxies(unsigned int n) const
{
	if (n<this->proxysets.size())
			return this->proxysets[n];
	return std::vector<T>();
}

}

#endif
