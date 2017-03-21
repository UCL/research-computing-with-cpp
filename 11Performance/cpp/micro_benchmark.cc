#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <time.h>
#include <vector>
#include <Eigen/Dense>

// Order for visiting the cities
typedef std::vector<Eigen::Index> Candidate;

// Evaluates travel distance for a given city order
template <class REAL = double> class TravelDistance {
public:
  typedef Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic> Coordinates;

  TravelDistance(Coordinates const &cities) : cities(cities) {}

  REAL operator()(Candidate const &candidate) const;
  Eigen::Index size() const { return cities.cols(); }

protected:
  Coordinates cities;
};

template <class REAL>
REAL TravelDistance<REAL>::operator()(Candidate const &candidate) const {
  if(candidate.size() != size())
    throw std::runtime_error("Candidate and cities sizes do not match");

  REAL result(0);
  for(Candidate::size_type i(1); i < size(); ++i)
    result += (cities.col(candidate[i]) - cities.col(candidate[i - 1])).norm();
  return result;
}

int main(int, char **) {
  auto const seed = 10;
  auto const warmup = 10000;
  auto const iterations = 1000000;
  auto const Nrows = 8;
  auto const Ncols = 500;
  std::cout << "Problem size " << Nrows << "x" << Ncols << "\n";

  // create random candidate
  Candidate candidate(Ncols, 0);
  std::iota(candidate.begin(), candidate.end(), 0);
  std::mt19937 rand_engine(seed);
  std::shuffle(candidate.begin(), candidate.end(), rand_engine);

  srand(seed);
  TravelDistance<double>::Coordinates const coordinates
      = TravelDistance<double>::Coordinates::Random(Nrows, Ncols) * 10;
  {
    // Perform warm-up so procs are ready
    TravelDistance<double> td_double(coordinates);
    for(auto i = 0; i < warmup; ++i)
      td_double(candidate);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for(auto i = 0; i < iterations; ++i)
      td_double(candidate);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_seconds
        = std::chrono::duration_cast<std::chrono::duration<double>>(end -
        start)
              .count();
    std::cout << "TravelDistance<double>: " << elapsed_seconds << " / "
              << iterations << " = "
              << elapsed_seconds / static_cast<double>(iterations) << "s\n";
  }

  {
    // Perform warm-up so procs are ready
    TravelDistance<float> td_float(coordinates.cast<float>());
    for(auto i = 0; i < warmup; ++i)
      td_float(candidate);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for(auto i = 0; i < iterations; ++i)
      td_float(candidate);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_seconds
        = std::chrono::duration_cast<std::chrono::duration<double>>(end -
        start)
              .count();
    std::cout << "TravelDistance<float>: " << elapsed_seconds << " / "
              << iterations << " = "
              << elapsed_seconds / static_cast<double>(iterations) << "s\n";
  }

  return 0;
}
