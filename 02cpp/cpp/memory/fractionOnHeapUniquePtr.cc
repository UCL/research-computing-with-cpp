#include "Fraction.h"
#include <memory>
int main() {
  // Don't do this... see code later on.
  std::unique_ptr<Fraction> f(new Fraction(1,4));
}

