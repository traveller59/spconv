// nvcc (cuda) 9.0 with gcc 5.5 don't support random, so compile it in host

#include <random>

namespace cuhash {

std::random_device random_dev;

std::mt19937 random_engine(random_dev());
std::uniform_int_distribution<unsigned> uint_distribution;

unsigned generate_random_uint32() { return uint_distribution(random_engine); }

} // namespace cuhash