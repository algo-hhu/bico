#include <Python.h>

#include "point/l2metric.h"
#include "point/squaredl2metric.h"
#include "point/point.h"
#include "point/pointweightmodifier.h"
#include "clustering/bico.h"
#include "datastructure/proxysolution.h"
#include "point/pointcentroid.h"

typedef unsigned int uint;

using namespace CluE;

class BicoExternal
{
public:
    BicoExternal(
        uint d,
        uint k,
        uint p,
        uint m,
        int seed);
    virtual ~BicoExternal();
    void addData(double const *array, uint n);
    void addPoint(double const *array);
    size_t compute(double *sample_weights,
                double *points);

private:
    const uint _d;
    Bico<Point> *_bico;
};

BicoExternal::BicoExternal(uint d,
                           uint k,
                           uint p,
                           uint m,
                           int seed) : _d(d), _bico(new Bico<Point>(d, k, p, m, seed, new SquaredL2Metric(), new PointWeightModifier()))
{}

void BicoExternal::addData(double const *array, uint n)
{
    for (size_t i = 0; i < n * _d; i += _d)
    {
        addPoint(&array[i]);
    }
}

void BicoExternal::addPoint(double const *array)
{
    std::vector<double> coords(array, array + _d);
    Point p(coords);
    // Call BICO point update
    *_bico << p;
}

size_t BicoExternal::compute(double *sample_weights,
                          double *points)
{
    // Retrieve coreset
    ProxySolution<Point> *sol = _bico->compute();
    // Output coreset points
    for (size_t i = 0; i < sol->proxysets[0].size(); ++i)
    {
        // Output weight
        sample_weights[i] = sol->proxysets[0][i].getWeight();
        // Output center of gravity
        for (size_t j = 0; j < sol->proxysets[0][i].dimension(); ++j)
        {
            points[i * _d + j] = sol->proxysets[0][i][j];
        }
    }
    size_t m = sol->proxysets[0].size();
    delete sol;

    return m;
}

BicoExternal::~BicoExternal()
{
    delete _bico;
}

// Thank you https://github.com/dstein64/kmeans1d!

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    BicoExternal *
    init(uint d,
         uint k,
         uint p,
         uint m,
         int seed)
    {
        return new BicoExternal(d, k, p, m, seed);
    }

#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    void addData(BicoExternal *bico, double const *array, uint n) { bico->addData(array, n); }

#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    void addPoint(BicoExternal *bico, double const *array) { bico->addPoint(array); }

#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    size_t compute(BicoExternal *bico, double *sample_weights,
                double *points) { return bico->compute(sample_weights, points); }

#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    void freeBico(BicoExternal *bico) {
        delete bico;
    }
} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "bico._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&_coremodule);
}
