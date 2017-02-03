#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>

/// comparator
class compMass {
public:
  compMass() :
    m_particlesOrdered({"neutrino", "electron", "muon", "pion", "kaon", "proton"})
  {};
  bool operator() (const std::string& s1, const std::string& s2) const
  {
    int index1=-1, index2=-1;
    for (int i=0;i<m_particlesOrdered.size();++i) {
      if (m_particlesOrdered[i]==s1) index1 = i;
      if (m_particlesOrdered[i]==s2) index2 = i;
    }
    return index1<index2; // unknown particles appear first (index=-1)
  }
private:
  std::vector<std::string> m_particlesOrdered;
};


/// main
int main()
{
  std::ifstream ifs("02stl/cpp/particleList.txt",std::ifstream::in);

  // Read in the data
  std::set<std::string,compMass> theParticles;
  std::string name;
  double momentum;
  while (!ifs.eof()) {
    ifs >> name >> momentum;
    if (!ifs.eof()) theParticles.insert(name);
  }
  ifs.close();

  // Output - it's already sorted!
  std::set<std::string>::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << *iter << std::endl;
  }      
}
