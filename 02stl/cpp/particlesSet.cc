#include <stdio.h>
#include <vector>
#include <set>
#include <string>
#include <iostream>

/// comparator
class compMass {
public:
  bool operator() (std::string s1, std::string s2)
  {
    int index1=-1, index2=-1;
    for (int i=0;i<m_particlesOrdered.size();++i) {
      if (m_particlesOrdered[i]==s1) index1 = i;
      if (m_particlesOrdered[i]==s2) index2 = i;
    }
    return index1<index2; // unknown particles appear first (index=-1)
  }
private:
  const std::vector<std::string> m_particlesOrdered =
    {"neutrino", "electron", "muon", "pion", "kaon", "proton"};
};


/// main
int main()
{
  FILE* ifp = fopen("02stl/cpp/particleList.txt","r");

  // Read in the data
  std::set<std::string,compMass> theParticles;
  char name[80];
  while (!feof(ifp)) {
    fscanf(ifp, "%s %*f", name);
    if (!feof(ifp)) theParticles.insert( std::string(name) );
  }
  fclose(ifp);

  // Output - it's already sorted!
  std::set<std::string>::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << *iter << std::endl;
  }      
}
