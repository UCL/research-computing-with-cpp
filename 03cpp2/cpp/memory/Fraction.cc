#include "Fraction.h"
#include <iostream>

Fraction::Fraction(int n, int d)
{
  numerator = n;
  denominator = d;
}

Fraction::~Fraction()
{
  std::cout << "I'm being deleted" << std::endl;
}

