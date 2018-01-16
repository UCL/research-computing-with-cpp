#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

int main()
{
  std::ifstream if1("03stl/cpp/randomNumbers1.txt",std::ifstream::in);
  std::ifstream if2("03stl/cpp/randomNumbers2.txt",std::ifstream::in);

  // Read in the data.
  int number;
  std::vector<int> theArray;
  while (!if1.eof()) {
    if1 >> number;
    if (!if1.eof()) theArray.push_back(number);
  }
  while (!if2.eof()) {
    if2 >> number;
    if (!if2.eof()) theArray.push_back(number);
  }
  if1.close();
  if2.close();

  // Sort and output
  std::sort(theArray.begin(),theArray.end());
  for (int i=0;i<theArray.size();++i) {
    std::cout << theArray[i] << " ";
  }      
}
