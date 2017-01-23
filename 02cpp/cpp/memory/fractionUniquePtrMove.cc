#include "Fraction.h"
#include <memory>
#include <iostream>

int main() {
  // Don't do this... see code later on.
  std::unique_ptr<Fraction> f(new Fraction(1,4));
  // std::unique_ptr<Fraction> f2 = f; // compile error

  std::cerr << "f=" << f << std::endl;

  std::unique_ptr<Fraction> f2;
  // f2 = f; // compile error

  f2.reset(f.get());

  std::cerr << "f=" << f << ", f2=" << f2 << std::endl;

}

