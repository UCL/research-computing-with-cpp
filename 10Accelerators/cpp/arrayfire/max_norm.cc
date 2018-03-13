#define CATCH_CONFIG_MAIN
#include <array>
#include <arrayfire.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <catch/catch.hpp>

/// "naive"
double naive(std::vector<std::array<double, 3>> const &Rs,
             std::array<double, 3> const &A) {
  double result(0);
  for(auto const &R : Rs) {
    auto norm = (A[0] - R[0]) * (A[0] - R[0]) + (A[1] - R[1]) * (A[1] - R[1])
                + (A[2] - R[2]) * (A[2] - R[2]);
    if(result < norm)
      result = norm;
  }
  return std::sqrt(result);
}

/// "broadcasting"
double broadcasting(std::vector<std::array<double, 3>> const &Rs,
                    std::array<double, 3> const &A) {
  af::array A_3x1(A.size(), A.data());
  af::array gpuRs(A.size(), Rs.size(), Rs[0].data());

  auto const A_3xn = af::tile(A_3x1, 1, Rs.size());
  auto const norms = af::sum(af::pow(gpuRs - A_3xn, 2), 0);
  auto const result = af::max(norms);
  std::shared_ptr<double> const host_array(result.host<double>(),
                                           [](double *ptr) { delete[] ptr; });
  return std::sqrt(*host_array);
}

/// "other"
double not_simplified(std::vector<std::array<double, 3>> const &Rs,
                      std::array<double, 3> const &A) {
  /// "transfer to gpu"
  af::array gpuRs(A.size(), Rs.size(), Rs[0].data());
  af::array gpuA(A.size(), A.data());

  /// "compute"
  auto const result
      = af::max(af::sum(af::pow(gpuRs - af::tile(gpuA, 1, Rs.size()), 2), 0));

  /// "transfer to cpu"
  std::shared_ptr<double> const host_array(result.host<double>(),
                                           [](double *ptr) { delete[] ptr; });

  /// "result"
  return std::sqrt(*host_array);
}

double broadcast_exercise(std::vector<std::array<double, 3>> const &Rs,
                          std::vector<std::array<double, 3>> const &Ks) {

  auto const n = Rs.size();
  auto const d = Rs[0].size();
  af::array gpuRs(d, n, Rs[0].data());
  af::array gpuKs(d, Ks.size(), Ks[0].data());

  auto const R0 = af::moddims(gpuRs, d, 1, n);
  auto const diff = af::tile(gpuRs, 1, 1, n) - af::tile(R0, 1, n, 1);

  // sum<double> does the sum and transfers the result back to host
  return af::sum<double>(
      af::cos(af::matmul(af::transpose(af::moddims(diff, d, n * n)), gpuKs)));
}

std::vector<bool> line_of_sight(std::vector<double> const &altitudes,
                                unsigned nOrientations, unsigned nDistances,
                                double stepsize=1) {
  af::array Zs(nDistances, nOrientations, altitudes.data());
  af::array steps = af::range(nDistances) * stepsize;

  auto const angles = af::atan2(Zs, af::tile(steps, 1, nOrientations));
  auto const result
      = (angles >= af::scan(angles, 0, AF_BINARY_MAX, false)).as(b8);

  std::shared_ptr<char> const host_array(result.host<char>(),
                                         [](char *ptr) { delete[] ptr; });
  return std::vector<bool>(host_array.get(),
                           host_array.get() + nOrientations * nDistances);
}

TEST_CASE("Naive vs arrayfire") {
  af::setBackend(AF_BACKEND_OPENCL);
  std::vector<std::array<double, 3>> const Rs
      = {{{1, 2, 3}}, {{4, 5, 6}}, {{0, 0, 7}}, {{1, 4, 8}}};
  std::array<double, 3> const A{{0, 0, 1}};

  CHECK(naive(Rs, A) == Approx(broadcasting(Rs, A)));
  CHECK(naive(Rs, A) == Approx(not_simplified(Rs, A)));
}

TEST_CASE("Broadcasting exercise") {
  af::setBackend(AF_BACKEND_CPU);
  std::vector<std::array<double, 3>> const Rs
      = {{{1, 2, 3}}, {{4, 5, 6}}, {{0, 0, 7}}, {{1, 4, 8}}};
  std::vector<std::array<double, 3>> const Ks = {{{2, 1, 1}}, {{1, -2, 6}}};

  double result = 0;
  for(auto const &R_i : Rs)
    for(auto const &R_j : Rs)
      for(auto const &K_k : Ks)
        result
            += std::cos(K_k[0] * (R_i[0] - R_j[0]) + K_k[1] * (R_i[1] - R_j[1])
                        + K_k[2] * (R_i[2] - R_j[2]));

  CHECK(broadcast_exercise(Rs, Ks) == Approx(result));
}

TEST_CASE("Line of sight") {
  af::setBackend(AF_BACKEND_CPU);
  auto const nOrientations = 2;
  auto const nDistances = 6;
  std::vector<double> const altitudes{
     0, 1, 2, 3,   2, 3, /* */
     0, 1, 2, 2.5, 2, 5,
  };
  std::vector<bool> const expected{true, true, true, true, false, false, /* */
                                   true, true, true, false, false, true};
  auto const actual = line_of_sight(altitudes, nOrientations, nDistances);
  CHECK(std::equal(expected.begin(), expected.end(), actual.begin()));
}
