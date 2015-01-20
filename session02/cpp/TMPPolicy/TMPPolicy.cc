#include <iostream>
#include <vector>
#include <stdexcept>

class NoRangeCheckingPolicy {
  public:
    static void CheckRange(size_t pos, size_t n) { return; } // no checking
};

class ThrowErrorRangeCheckingPolicy {
  public:
    static void CheckRange(size_t pos, size_t n)
    {
      if (pos >= n) { throw std::runtime_error("Out of range!"); }
    }
};

template < typename T
         , typename RangeCheckingPolicy = NoRangeCheckingPolicy
         >
class Vector
 : public RangeCheckingPolicy
{

  private:
    std::vector<T> data;
  public:
    // other methods etc.
    const T& operator[] (size_t pos) const
    {
      RangeCheckingPolicy::CheckRange(pos, data.size());
      return data[pos];
    }
};

int main () {
  Vector<int, ThrowErrorRangeCheckingPolicy> a;
  // a.push_back(1); or similar
  // a.push_back(2); or similar
  try {
    std::cout << a[3] << std::endl;
  } catch (const std::runtime_error& e)
  {
    std::cerr << e.what();
  }
  return 0;
}
