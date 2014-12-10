#include "fraction.h"
Fraction::Fraction()
{
  m_Numerator = 0;
  m_Denominator = 1;
}

Fraction::Fraction(const int& num, const int& denom)
{
  m_Numerator = num;
  m_Denominator = denom;
}

Fraction::~Fraction()
{
  // Nothing to do. int will destroy itself correctly.
}

int Fraction::numerator() const
{
  return m_Numerator;
}

int Fraction::denominator() const
{
  return m_Denominator;
}

Fraction& Fraction::operator+(const Fraction& another)
{
  // Should be simplified.
  int denominator = m_Denominator*another.m_Denominator;
  int numerator = m_Numerator*another.denominator() + another.numerator()*m_Denominator;
  m_Denominator = denominator;
  m_Numerator = numerator;
  return *this;
}
