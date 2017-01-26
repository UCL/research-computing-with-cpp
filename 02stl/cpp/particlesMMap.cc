#include <stdio.h>
#include <map>
#include <string>
#include <iostream>

int main()
{
  FILE* ifp = fopen("02stl/cpp/particleList.txt","r");

  // Read in the data
  std::multimap<std::string,float> theParticles;
  char name[80];
  float momentum;
  while (!feof(ifp)) {
    fscanf(ifp, "%s %f", name, &momentum);
    if (!feof(ifp)) theParticles.insert( std::pair<std::string,float>(std::string(name),momentum) );
  }
  fclose(ifp);

  // Output - it's already sorted!
  std::multimap<std::string,float>::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << (*iter).first << " " << (*iter).second << std::endl;
  }      
}
