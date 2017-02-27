#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <cmath>
#include "smooth.h"

TEST_CASE("Smooth model can be instantiated and configured", "[Smooth]") {

  SECTION("Smooth can be constructed") {
    Smooth smooth;
    REQUIRE(smooth.Size() == 10000);
    REQUIRE(smooth.Field().size() == 100);
    REQUIRE(smooth.Field()[0].size() == 100);
  }
}

TEST_CASE("Smooth mathematical functions are correct", "[Smooth]") {
  Smooth smooth;
  SECTION("Disk support function is correct") {
    REQUIRE(smooth.Disk(500) == 0.0);
    REQUIRE(smooth.Disk(21.6) == 0.0);
    REQUIRE(smooth.Disk(21.4) > 0.0);
    REQUIRE(smooth.Disk(21.4) < 1.0);
    REQUIRE(smooth.Disk(20.6) > 0.0);
    REQUIRE(smooth.Disk(20.6) < 1.0);
    REQUIRE(smooth.Disk(20.4) == 1.0);
    REQUIRE(smooth.Disk(19.0) == 1.0);
    REQUIRE(smooth.Disk(21.0) == 0.5);
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

  /// End
  SECTION("Transition function is correct") {
    REQUIRE(abs(smooth.transition(1.0, 0.3) - 1.0) < 0.1);
    REQUIRE(smooth.transition(1.0, 1.0) == 0.0);
    REQUIRE(abs(smooth.transition(0.0, 0.3) - 1.0) < 0.1);
    REQUIRE(abs(smooth.transition(0.0, 0.0)) < 0.1);
  }
  SECTION("Wraparound Distance is correct") {
    REQUIRE(smooth.TorusDifference(95, 5, 100) == 10);
    REQUIRE(smooth.TorusDifference(5, 96, 100) == 9);
    REQUIRE(smooth.TorusDifference(5, 10, 100) == 5);
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
  SECTION("DiskFillingUnityWithDiskSeed") {
    smooth.SeedDisk();
    REQUIRE(abs(smooth.FillingDisk(0, 0) - 1.0) < 0.1);
  }

  SECTION("Disk Filling Zero With Ring Seed") {
    smooth.SeedRing();
    REQUIRE(abs(smooth.FillingDisk(0, 0)) < 0.1);
  }
  SECTION("RingFillingUnityWithRingSeed") {
    smooth.SeedRing();
    REQUIRE(abs(smooth.FillingRing(0, 0) - 1.0) < 0.1);
  }
}

TEST_CASE("FillingFieldHasRangeofValues") {
  Smooth smooth(300, 300);
  smooth.SeedRing();
  double min = 1.0;
  double max = 0.0;
  for(int x = 0; x < 300; x++) {
    double filling = smooth.FillingRing(x, 0);
    if(filling < min)
      min = filling;
    if(filling > max)
      max = filling;
  }
  REQUIRE(min < 0.2);
  REQUIRE(max > 0.4);
}
