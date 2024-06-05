#include "../misc/randomness.h"

using namespace CluE;

std::mt19937 Randomness::mt19937Generator(static_cast<unsigned int>(time(0)));
