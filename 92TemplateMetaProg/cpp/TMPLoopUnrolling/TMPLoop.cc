#include <iostream>
#include <vector>
using namespace std;

template<typename T>
T Sum(const std::vector<T>& data)
{
  T total = 0;
  for (size_t i = 0; i < data.size(); i++)
  {
    total += data[i];
  }
  return total;
}

int main () {
  size_t numberOfInts = 3;
  size_t numberOfLoops = 1000000000;
  vector<int> a(numberOfInts);
  int total = 0;

  std::cout << "Started" << std::endl;
  for (size_t j = 0; j < numberOfLoops; j++)
  {
    for (size_t i = 0; i < numberOfInts; i++)
    {
      total = Sum(a);
    }
  }
  std::cout << "Finished:" << total << std::endl;
}
