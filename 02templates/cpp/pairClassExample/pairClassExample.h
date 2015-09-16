template <typename T>
class MyPair {
    T m_Values [2];
  public:
    MyPair(const T& first, const T& second);
    T getMax() const;
};
#include "pairClassExample.cc"