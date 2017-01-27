#include <stdio.h>
#include <map>
#include <string>
#include <iostream>

int main()
{
  FILE* ifp = fopen("02stl/cpp/particleList.txt","r");

  // Read in the data
  std::multimap<std::string,double> theParticles;
  char name[80];
  double momentum;
  while (!feof(ifp)) {
    fscanf(ifp, "%s %f", name, &momentum);
    if (!feof(ifp)) theParticles.insert( std::pair<std::string,double>(std::string(name),momentum) );
  }
  fclose(ifp);

  // Output - it's already sorted!
  std::multimap<std::string,double>::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << (*iter).first << " " << (*iter).second << std::endl;
  }      
}
