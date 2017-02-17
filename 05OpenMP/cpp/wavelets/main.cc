#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "wavelets.h"
#include <algorithm>
#include <catch/catch.hpp>

TEST_CASE("Application of Cyclical Filters")
{
  wavelets::Signal const signal{ 1, 2, 3, 5, 5, 6 };
  wavelets::Signal const filter{ 3, -3 };

  SECTION("No cyclical overlap problem")
  {
    CHECK(wavelets::apply_cyclical_filter(signal, 0, filter) == Approx(-3));
    CHECK(wavelets::apply_cyclical_filter(signal, 1, filter) == Approx(-3));
    CHECK(wavelets::apply_cyclical_filter(signal, 2, filter) == Approx(-6));
  }
  SECTION("Filter reaches exactly end of signal")
  {
    CHECK(wavelets::apply_cyclical_filter(signal, 4, filter) == Approx(-3));
  }
  SECTION("Filter reaches past end of signal")
  {
    CHECK(wavelets::apply_cyclical_filter(signal, 5, filter) == Approx(15));
  }

  SECTION("Empty filter")
  {
    CHECK(wavelets::apply_cyclical_filter({ 5, 2 }, 0, {}) == Approx(0));
  }
  SECTION("Empty signal")
  {
    CHECK(wavelets::apply_cyclical_filter({}, 0, { 1, 2 }) == Approx(0));
  }
  SECTION("Empty signal and filter")
  {
    CHECK(wavelets::apply_cyclical_filter({}, 0, {}) == Approx(0));
  }
  SECTION("Filter much much bigger than signal")
  {
    CHECK(wavelets::apply_cyclical_filter({ 1, 2 }, 0, { 1, 2, 3, 4, 5 }) == Approx(21));
  }
}

TEST_CASE("Single pass wavelet transform")
{
  // Use unnormalized haar wavelets, because simple
  // Normalize by multiplyint the coeffs by std::sqrt(2)
  wavelets::DaubechyData const haar{ { 1, 1 }, { 1, -1 } };
  SECTION("Haar")
  {
    // Odd number of elements to test periodicity
    auto const actual = single_direct_transform({ 1, 2, 3, 5, 5, 6, 8 }, haar);
    //                                 high pass             low pass
    //                                     =                     =
    //                                  details            approximation
    wavelets::Signal const expected{ -1, -2, -1, 7, /*   */ 3, 8, 11, 9 };
    REQUIRE(actual.size() == expected.size());
    CHECK(std::equal(actual.begin(), actual.end(), expected.begin()));
  }

  SECTION("Empty signal")
  {
    CHECK(wavelets::single_direct_transform({}, haar).size() == 0);
  }
}

TEST_CASE("Multi-pass wavelet transform")
{
  wavelets::DaubechyData const haar{ { 1, 1 }, { 1, -1 } };
  SECTION("Empty signal")
  {
    CHECK(wavelets::direct_transform({}, haar, 2).size() == 0);
  }
  SECTION("Level 0 is a copy")
  {
    wavelets::Signal const signal{ 1, 2, 3, 4 };
    auto const actual = wavelets::direct_transform(signal, haar, 0);
    REQUIRE(actual.size() == signal.size());
    CHECK(std::equal(actual.begin(), actual.end(), signal.begin()));
  }
  SECTION("Haar")
  {
    wavelets::Signal const signal{ 1, 2, 3, 5, 5, 6, 8 };
    std::vector<wavelets::Signal> const expecteds{
      signal,                                     // level 0
      { -1, -2, -1, 7, 3, 8, 11, 9 },             // level 1
      { -1, -2, -1, 7, -5, 2, 11, 20 },           // level 2
      { -1, -2, -1, 7, -5, 2, -9, 31 },           // level 3
      { -1, -2, -1, 7, -5, 2, -9, 0, 62 },        // level 4
      { -1, -2, -1, 7, -5, 2, -9, 0, 0, 124 },    // level 5
      { -1, -2, -1, 7, -5, 2, -9, 0, 0, 0, 248 }, // level 6
    };
    for (decltype(expecteds.size()) levels(0); levels < expecteds.size(); ++levels) {
      auto const actual = direct_transform(signal, haar, levels);
      REQUIRE(actual.size() == expecteds[levels].size());
      CHECK(std::equal(actual.begin(), actual.end(), expecteds[levels].begin()));
    }
  }
}
