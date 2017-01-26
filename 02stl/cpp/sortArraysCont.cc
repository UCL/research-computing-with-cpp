#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iostream>

int main()
{
  FILE* if1 = fopen("02stl/cpp/randomNumbers1.txt","r");
  FILE* if2 = fopen("02stl/cpp/randomNumbers2.txt","r");

  // Read in the data.
  int number;
  std::vector<int> theArray;
  while (!feof(if1)) {
    fscanf(if1, "%d",&number);
    if (!feof(if1)) theArray.push_back(number);
  }
  while (!feof(if2)) {
    fscanf(if2, "%d",&number);
    if (!feof(if2)) theArray.push_back(number);
  }
  fclose(if1);
  fclose(if2);

  // Sort and output
  std::sort(theArray.begin(),theArray.end());
  for (int i=0;i<theArray.size();++i) {
    std::cout << theArray[i] << " ";
  }      
}
