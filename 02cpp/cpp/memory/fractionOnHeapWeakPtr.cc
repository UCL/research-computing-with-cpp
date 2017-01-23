#include "Fraction.h"
#include <memory>
#include <iostream>
int main() {
  std::shared_ptr<Fraction> s1(new Fraction(1,4));
  std::weak_ptr<Fraction> w1;      // can point to nothing
  std::weak_ptr<Fraction> w2 = s1; // assignment from shared
  std::weak_ptr<Fraction> w3(s1);  // construction from shared

  // Can't be de-referenced!!!
  // std::cerr << "Value=" << w1->numerator << "/" << w1->denominator << std::endl;

  // Needs converting to shared, and checking
  std::shared_ptr<Fraction> s2 = w1.lock();
  if (s2)
  {
    std::cerr << "Object w1 exists=" << s2->numerator << "/" << s2->denominator << std::endl;
  }

  // Or, create shared, check for exception
  std::shared_ptr<Fraction> s3(w2);
  std::cerr << "Object must exists=" << s3->numerator << "/" << s3->denominator << std::endl;
}

