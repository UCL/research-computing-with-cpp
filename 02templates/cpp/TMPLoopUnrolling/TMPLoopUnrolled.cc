#include <iostream>
#include <vector>
using namespace std;

template <int length>
class FixedIntVector {
   std::vector<int> data;
  public:
    FixedIntVector()
    {
      data.resize(length);
    }
    int Sum()
    {
      int sum = 0;
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
  FixedIntVector<numberOfInts> a;
  int total = 0;

  std::cout << "Started" << std::endl;
  for (size_t j = 0; j < numberOfLoops; j++)
  {
    total = a.Sum();
  }
  std::cout << "Finished:" << total << std::endl;
}
