#include <map>
#include <string>
#include <iostream>
#include <fstream>

int main()
{
  std::ifstream ifs("02stl/cpp/particleList.txt",std::ifstream::in);
  // Read in the data
  std::multimap<std::string,double> theParticles;
  std::string name;
  double momentum;
  while (!ifs.eof()) {
    ifs >> name >> momentum;
    if (!ifs.eof())
      theParticles.insert( std::pair<std::string,double>(std::string(name),momentum) );
  }
  ifs.close();
  // Output - it's already sorted!
  std::multimap<std::string,double>::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << iter->first << " " << iter->second << std::endl;
  }      
}
