#ifndef HAS_MPI
#define CATCH_CONFIG_MAIN
#else
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>
#endif

#include <cmath>
#include <random>
#include "catch.hpp"
#include "smooth.h"

TEST_CASE("Compute Integrals") {
  Smooth smooth(300, 300);
  smooth.SeedConstant(0);

  // check for different positions in the torus
  for(auto const x : {150, 298, 0})
    for(auto const y : {150, 298, 0}) {
      SECTION("At position (" + std::to_string(x) + ", " + std::to_string(y) + ")") {
        SECTION("Ring only") {
          smooth.AddRing(150, 150);

          auto const result = smooth.Integrals(150, 150);
          // 0.1 accuracy because of smoothing
          CHECK(std::get<0>(result) == Approx(0).epsilon(0.1));
          CHECK(std::get<1>(result) == Approx(1).epsilon(0.1));
        }

        SECTION("Disk only") {
          smooth.AddDisk(150, 150);
          auto const result = smooth.Integrals(150, 150);
          CHECK(std::get<0>(result) == Approx(1).epsilon(0.1));
          CHECK(std::get<1>(result) == Approx(0).epsilon(0.1));
        }

        SECTION("Disk and ring") {
          smooth.AddRing(150, 150);
          smooth.AddDisk(150, 150);
          auto const result = smooth.Integrals(150, 150);
          CHECK(std::get<0>(result) == Approx(1).epsilon(0.1));
          CHECK(std::get<1>(result) == Approx(1).epsilon(0.1));
        }
      }
    }
}

TEST_CASE("Update") {
  // just test playing with a single pixel lit up sufficiently that the
  // transition is non-zero in the ring.
  auto const radius = 5;
  Smooth smooth(100, 100, radius);
  smooth.AddPixel(50, 50, 0.3 * smooth.NormalisationRing());
  CHECK(std::get<0>(smooth.Integrals(50, 50))
        == Approx(0.3 * smooth.NormalisationRing() / smooth.NormalisationDisk()));

  // check the integrals are numbers for which Transition gives non-zero result
  // in the ring
  CHECK(std::get<1>(smooth.Integrals(50, 50)) == Approx(0));
  CHECK(std::get<0>(smooth.Integrals(40, 40)) == Approx(0));
  CHECK(std::get<1>(smooth.Integrals(40, 40)) == Approx(0.3));
  CHECK(std::get<0>(smooth.Integrals(42, 39)) == Approx(0));
  CHECK(std::get<1>(smooth.Integrals(42, 39)) == Approx(0.3));

  // Now call update
  smooth.Update();
  auto const field = smooth.Field();
  // And check death in the disk
  CHECK(field[smooth.Index(50, 50)] == Approx(0));
  CHECK(field[smooth.Index(51, 52)] == Approx(0));
  // And check life in the ring
  CHECK(field[smooth.Index(45, 45)] == Approx(smooth.Transition(0, 0.3)));
  CHECK(field[smooth.Index(42, 39)] == Approx(smooth.Transition(0, 0.3)));
  // And check death outside
  CHECK(field[smooth.Index(15, 15)] == Approx(0));
}

#ifdef HAS_MPI
TEST_CASE("Arithmetics for plitting a field on different nodes") {
  CHECK(Smooth::OwnedStart(5, 2, 0) == 0);
  CHECK(Smooth::OwnedStart(5, 2, 1) == 3);

  for(int i(0); i < 5; ++i)
    CHECK(Smooth::OwnedStart(5, 5, i) == i);

  // with too many procs, some procs have empty ranges
  for(int i(5); i < 10; ++i)
    CHECK(Smooth::OwnedStart(5, 10, i) == 5);
}

TEST_CASE("Sync whole field") {
  int rank, ncomms;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ncomms);

  // Create known field: -1 outside owned range, equal to rank inside
  // Different on each process!
  // Also, we make sure the size does not split evenly with the number of procs,
  // because that is a harder test.
  std::vector<density> field(5 * ncomms + ncomms / 3, -1);
  std::fill(field.begin() + Smooth::OwnedStart(field.size(), ncomms, rank),
            field.begin() + Smooth::OwnedStart(field.size(), ncomms, rank + 1), rank);

  SECTION("Blocking synchronisation") {
    Smooth::WholeFieldBlockingSync(field, MPI_COMM_WORLD);

    for(int r(0); r < ncomms; ++r)
      CHECK(std::all_of(field.begin() + Smooth::OwnedStart(field.size(), ncomms, r),
                        field.begin() + Smooth::OwnedStart(field.size(), ncomms, r + 1),
                        [r](density d) { return std::abs(d - r) < 1e-8; }));
  }

  SECTION("Non blocking synchronisation") {
    auto request = Smooth::WholeFieldNonBlockingSync(field, MPI_COMM_WORLD);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    for(int r(0); r < ncomms; ++r)
      CHECK(std::all_of(field.begin() + Smooth::OwnedStart(field.size(), ncomms, r),
                        field.begin() + Smooth::OwnedStart(field.size(), ncomms, r + 1),
                        [r](density d) { return std::abs(d - r) < 1e-8; }));
  }
}

TEST_CASE("Serial vs parallel") {
  Smooth serial(100, 100, 5);
  Smooth parallel(100, 100, 5);
  parallel.Communicator(MPI_COMM_WORLD);

  // generate one field for all Smooth instances
  std::vector<density> field(100 * 100);
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> randdist(0, 1);
  std::generate(field.begin(), field.end(), [&randdist, &gen]() { return randdist(gen); });
  MPI_Bcast(field.data(), field.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // set the fields for both Smooth instances
  serial.Field(field);
  parallel.Field(field);

  int rank, ncomms;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ncomms);

  auto const start = Smooth::OwnedStart(field.size(), ncomms, rank);
  auto const end = Smooth::OwnedStart(field.size(), ncomms, rank);
  // if this is false, then the test itself is wrong
  CHECK(std::equal(serial.Field().begin() + start, serial.Field().begin() + end,
                   parallel.Field().begin() + start));

  SECTION("Blocking synchronization") {
    // check fields are the same in parallel and in serial for a few iterations
    for(int i(0); i < 3; ++i) {
      serial.Update();
      parallel.Update();
      CHECK(std::equal(serial.Field().begin() + start, serial.Field().begin() + end,
                       parallel.Field().begin() + start));
    }
  }

  SECTION("Layered communication-computation") {
    for(int i(0); i < 3; ++i) {
      serial.Update();
      parallel.LayeredUpdate();
      CHECK(std::equal(serial.Field().begin() + start, serial.Field().begin() + end,
                       parallel.Field().begin() + start));
    }
  }
}

TEST_CASE("Smooth model can be instantiated and configured", "[Smooth]") {

  SECTION("Smooth can be constructed") {
    Smooth smooth;
    REQUIRE(smooth.Size() == 10000);
    REQUIRE(smooth.Field().size() == smooth.Size());
  }
}

TEST_CASE("Smooth mathematical functions are correct", "[Smooth]") {
  Smooth smooth;
  SECTION("Disk support function is correct") {
    REQUIRE(smooth.Disk(500) == Approx(0));
    REQUIRE(smooth.Disk(21.6) == 0.0);
    REQUIRE(smooth.Disk(21.4) > 0.0);
    REQUIRE(smooth.Disk(21.4) < 1.0);
    REQUIRE(smooth.Disk(20.6) > 0.0);
    REQUIRE(smooth.Disk(20.6) < 1.0);
    REQUIRE(smooth.Disk(20.4) == Approx(1.0));
    REQUIRE(smooth.Disk(19.0) == Approx(1.0));
    REQUIRE(smooth.Disk(21.0) == Approx(0.5));
  }
  SECTION("Ring support function is correct") {
    REQUIRE(smooth.Ring(22) == 1.0);
    REQUIRE(smooth.Ring(21.6) == 1.0);
    REQUIRE(smooth.Ring(21.4) > 0.0);
    REQUIRE(smooth.Ring(21.4) < 1.0);
    REQUIRE(smooth.Ring(20.6) > 0.0);
    REQUIRE(smooth.Ring(20.6) < 1.0);
    REQUIRE(smooth.Ring(20.4) == 0.0);
    REQUIRE(smooth.Ring(21.0) == 0.5);
    REQUIRE(smooth.Ring(64.0) == 0.0);
    REQUIRE(smooth.Ring(63.6) == 0.0);
    REQUIRE(smooth.Ring(63.4) > 0.0);
    REQUIRE(smooth.Ring(63.4) < 1.0);
    REQUIRE(smooth.Ring(62.6) > 0.0);
    REQUIRE(smooth.Ring(62.6) < 1.0);
    REQUIRE(smooth.Ring(62.4) == 1.0);
    REQUIRE(smooth.Ring(63.0) == 0.5);
  }

  /// Sigmoid_Test
  SECTION("Sigmoid function is correct") {
    double e = std::exp(1.0);
    REQUIRE(Smooth::Sigmoid(1.0, 1.0, 4.0) == 0.5);
    REQUIRE(std::abs(Smooth::Sigmoid(1.0, 0.0, 4.0) - e / (1 + e)) < 0.0001);
    REQUIRE(Smooth::Sigmoid(10000, 1.0, 4.0) == 1.0);
    REQUIRE(std::abs(Smooth::Sigmoid(0.0, 1.0, 0.1)) < 0.001);
  }
  /// end
  SECTION("Transition function is correct") {
    REQUIRE(std::abs(smooth.Transition(1.0, 0.3) - 1.0) < 0.1);
    REQUIRE(smooth.Transition(1.0, 1.0) == Approx(0));
    REQUIRE(std::abs(smooth.Transition(0.0, 0.3) - 1.0) < 0.1);
    REQUIRE(std::abs(smooth.Transition(0.0, 0.0)) < 0.1);
  }
  SECTION("Wraparound Distance is correct") {
    REQUIRE(smooth.TorusDistance(95, 5, 100) == 10);
    REQUIRE(smooth.TorusDistance(5, 96, 100) == 9);
    REQUIRE(smooth.TorusDistance(5, 10, 100) == 5);
    REQUIRE(smooth.Radius(10, 10, 13, 14) == 5.0);
  }
}

TEST_CASE("NormalisationsAreCorrect") {
  Smooth smooth(100, 100, 10);
  SECTION("Disk Normalisation is correct") {
    // Should be roughly pi*radius*radius,
    REQUIRE(std::abs(smooth.NormalisationDisk() - 314.15) < 1.0);
  }
  SECTION("Ring Normalisation is correct") {
    // Should be roughly pi*outer*outer-pi*inner*inner, pi*100*(9-1), 2513.27
    REQUIRE(std::abs(smooth.NormalisationRing() - 2513.27) < 2.0);
  }
}

TEST_CASE("FillingsAreUnityWhenSeeded") {
  Smooth smooth;
  smooth.SeedConstant(0);
  SECTION("DiskFillingUnityWithDiskSeed") {
    smooth.AddDisk();
    REQUIRE(std::get<0>(smooth.Integrals(0, 0)) == Approx(1).epsilon(0.1));
  }

  SECTION("Disk Filling Zero With Ring Seed") {
    smooth.AddRing();
    REQUIRE(std::get<0>(smooth.Integrals(0, 0)) == Approx(0).epsilon(0.1));
  }
  SECTION("RingFillingUnityWithRingSeed") {
    smooth.AddRing();
    REQUIRE(std::get<1>(smooth.Integrals(0, 0)) == Approx(1).epsilon(0.1));
  }
}

TEST_CASE("FillingFieldHasRangeofValues") {
  Smooth smooth(300, 300);
  smooth.SeedConstant(0);
  smooth.AddRing();
  double min = 1.0;
  double max = 0.0;
  for(int x = 0; x < 300; x++) {
    double filling = std::get<1>(smooth.Integrals(x, 0));
    min = std::min(min, filling);
    max = std::max(max, filling);
  }
  REQUIRE(min < 0.2);
  REQUIRE(max > 0.4);
}

int main(int argc, const char **argv) {
  // There must be exactly once instance
  Catch::Session session;

  MPI_Init(&argc, const_cast<char ***>(&argv));
  auto const result = session.run();

  MPI_Finalize();

  return result;
}
#endif
