#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include <Eigen/Dense>

// Underlying type for temperature, distance, energy
typedef double Real;
// Order for visiting the cities
typedef std::vector<Eigen::Index> Candidate;

// Evaluates travel distance for a given city order
class TravelDistance {
public:
  typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> Coordinates;

  TravelDistance(Coordinates const &cities) : cities(cities) {}

  Real operator()(Candidate const &candidate) const;
  Eigen::Index size() const { return cities.cols(); }

protected:
  Coordinates cities;
};

class SimulatedAnnealing {
public:
  typedef std::pair<Real, Candidate> Result;

  SimulatedAnnealing(TravelDistance const &travel_distance, Real temperature,
                     int itermax = 100, double variability = 0.25,
                     bool verbose = true)
      : travel_distance(travel_distance), temperature(temperature),
        itermax(itermax), variability(variability), verbose(verbose) {}
  SimulatedAnnealing(TravelDistance::Coordinates const &coordinates,
                     Real temperature)
      : SimulatedAnnealing(TravelDistance(coordinates), temperature) {}

  Result operator()(Candidate const &candidate) const;
  Result operator()() const {
    Candidate candidate(travel_distance.size(), 0);
    std::iota(candidate.begin(), candidate.end(), 0);
    return operator()(candidate);
  }

protected:
  TravelDistance travel_distance;
  Real temperature;
  int itermax;
  double variability;
  bool verbose;

  // Creates a neighbor to current
  static Candidate neighbor(TravelDistance const &travel_distance,
                            Candidate const &current,
                            double variability = 0.25);
  // Creates a neighbor to current
  Candidate neighbor(Candidate const &current) const {
    return neighbor(travel_distance, current, variability);
  }
  // True if candidate is better
  static bool
  compare(Real const &beta, Real const &current, Real const &candidate);
  // True if candidate is better
  bool compare(Real const &current, Real const &candidate) const {
    return compare(1e0 / temperature, current, candidate);
  }
  // Simple way to get a random number.
  // Not quite the right C++11 way to do it.
  static double rand_real() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  }
};

Real TravelDistance::operator()(Candidate const &candidate) const {
  if(candidate.size() != size())
    throw std::runtime_error("Candidate and cities sizes do not match");

  Real result(0);
  for(Candidate::size_type i(1); i < size(); ++i)
    result += (cities.col(candidate[i]) - cities.col(candidate[i - 1])).norm();
  return result;
}

std::string print_candidate(Candidate const &candidate, double energy) {
  std::ostringstream sstr;
  sstr << "(" << candidate[0];
  for(int i(1); i < candidate.size(); ++i)
    sstr << ", " << candidate[i];
  sstr << ") with energy " << energy;
  return sstr.str();
}
std::string print_candidate(SimulatedAnnealing::Result const &candidate) {
  return print_candidate(candidate.second, candidate.first);
}

// True if new candidate better
bool SimulatedAnnealing::compare(Real const &beta, Real const &minimum,
                                 Real const &candidate) {
  if(minimum > candidate)
    return true;
  return rand_real() < std::exp(-beta * (candidate - minimum));
}

Candidate SimulatedAnnealing::neighbor(TravelDistance const &travel_distance,
                                       Candidate const &current,
                                       double variability) {
  auto neighbor = current;
  auto const n = travel_distance.size();
  do {
    int const i = static_cast<int>(std::min<double>(rand_real() * n, n - 1));
    int const j = static_cast<int>(std::min<double>(rand_real() * n, n - 1));
    if(i != j)
      std::swap(neighbor[i], neighbor[j]);
  } while(SimulatedAnnealing::rand_real() > variability);
  return neighbor;
}

SimulatedAnnealing::Result SimulatedAnnealing::
operator()(Candidate const &initial_guess) const {
  SimulatedAnnealing::Result current{travel_distance(initial_guess),
                                     initial_guess};
  auto minimum = current;
  if(verbose)
    std::cout << "First candidate " << print_candidate(minimum) << "\n";
  for(int i(0); i < itermax and current.first > 1e-12; ++i) {
    Result candidate{0, neighbor(current.second)};
    candidate.first = travel_distance(candidate.second);
    if(compare(current.first, candidate.first)) {
      std::swap(current, candidate);
      if(verbose)
        std::cout << "Found candidate " << print_candidate(current) << "\n";
      if(current.first < minimum.first)
        minimum = current;
    }
  }
  return minimum;
}

int main(int, char **) {
  srand(time(NULL));
  TravelDistance::Coordinates coordinates
      = TravelDistance::Coordinates::Random(3, 50) * 10;
  SimulatedAnnealing const sa(
      coordinates,
      1,      // temperature
      1000,   // itermax
      0.25,   // variability, e.g. distance from current solution
      false); // whether to be verbose

  // solve using default input guess
  auto const result = sa();

  std::cout << "Final candidate " << print_candidate(result) << "\n";

  return 0;
}
