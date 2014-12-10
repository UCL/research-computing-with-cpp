// Defining a User Defined Type.
class Fraction {

public: // access control

  // How to create
  Fraction();
  Fraction(const int& num, const int& denom);

  // How to destroy
  ~Fraction();

  // How to access
  int numerator() const;
  int denominator() const;

  // What you can do
  Fraction& operator+(const Fraction& another);

private: // access control

  // The data
  int m_Numerator;
  int m_Denominator;
};