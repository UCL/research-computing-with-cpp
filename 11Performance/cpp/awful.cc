#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

struct Cities {
  // Number of cities
  int number;
  // All X coordinates
  double *x;
  // All Y coordinates
  double *y;
};

double evaluate(Cities cities, int *order) {
  double result(0);
  for(int i(2); i < cities.number; ++i) {
    assert(order[i] >= 0 and order[i] < cities.number);
    result += std::sqrt(
        std::pow(*(cities.x + order[i]) - *(cities.x + order[i - 1]), 2)
        + std::pow(*(cities.y + order[i]) - *(cities.y + order[i - 1]), 2));
  }
  return result;
}

double rand_real() {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

std::string print_candidate(int n, int *candidate, double energy) {
  std::ostringstream sstr;
  sstr << "(" << candidate[0];
  for(int i(1); i < n; ++i)
    sstr << ", " << candidate[i];
  sstr << ") with energy " << energy;
  return sstr.str();
}

// True if new candidate better
bool compare(double beta, double minimum, double candidate) {
  if(minimum > candidate)
    return true;
  return rand_real() < std::exp(-beta * (candidate - minimum));
}

int *neighbor(int n, int *order, double mutability = 0.25) {
  int *neighbor = new int[n];
  std::copy(order, order + n, neighbor);
  do {
    int const i = static_cast<int>(std::min<double>(rand_real() * n, n - 1));
    int const j = static_cast<int>(std::min<double>(rand_real() * n, n - 1));
    if(i != j)
      std::swap(*(neighbor + i), *(neighbor + j));
  } while(rand_real() > mutability);
  return neighbor;
}

int main(int, char **) {
  double x[] = {0, 1, 2, 3, 4};
  double y[] = {4, 3, -2, 5, 6};
  Cities const cities = {5, x, y};

  auto const itermax = 100;
  auto const temperature = 1;

  // Holds current location in optimization
  auto *current = new int[cities.number];
  std::iota(current, current + cities.number, 0);
  double e_current = evaluate(cities, current);

  // Holds minimum solution found
  auto *minimum = new int[cities.number];
  std::copy(current, current + cities.number, minimum);
  auto e_minimum = e_current;

  std::cout << "First candidate "
            << print_candidate(cities.number, current, e_current) << "\n";

  for(auto iter = 0; iter < itermax and e_current > 1e-12; ++iter) {

    int *candidate = neighbor(cities.number, current);
    auto const e_candidate = evaluate(cities, candidate);
    auto const do_swap = compare(1 / temperature, e_current, e_candidate);

    if(not do_swap)
      continue;

    std::swap(candidate, current);
    e_current = e_candidate;
    std::cout << "Found candidate "
              << print_candidate(cities.number, current, e_current) << "\n";

    // keep track of minimum
    if(e_minimum > e_current) {
      e_minimum = e_current;
      std::copy(current, current + cities.number, minimum);
    }
  }

  std::cout << "Final candidate "
            << print_candidate(cities.number, minimum, e_minimum) << "\n";

  delete[] minimum;
  delete[] current;
}
