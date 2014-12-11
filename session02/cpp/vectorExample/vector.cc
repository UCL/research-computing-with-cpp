#include <vector>
#include <iostream>
int main(int argc, char** argv)
{
  std::vector<int> v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  std::cout << v1.size() << std::endl;
  return 0;
}
