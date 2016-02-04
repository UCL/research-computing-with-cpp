#include <iostream>
#include <vector>
using namespace std;

template <typename T, int length>
class FixedVector {
   T data[length];
  public:
    FixedVector()
    {
      // Initialise
      for (size_t i = 0; i < length; i++)
      {
        data[i] = 0;
      }
    }
    T Sum()
    {
      T sum = 0;
      for (size_t i = 0; i < length; i++)
      {
        sum += data[i];
      }
      return sum;
    }
};

int main () {
  const size_t numberOfInts = 3;
  const size_t numberOfLoops = 1000000000;
  FixedVector<int, numberOfInts> a;
  int total = 0;

  std::cout << "Started" << std::endl;
  for (size_t j = 0; j < numberOfLoops; j++)
  {
    for (size_t i = 0; i < numberOfInts; i++)
    {
      total = a.Sum();
    }
  }
  std::cout << "Finished:" << total << std::endl;
}
