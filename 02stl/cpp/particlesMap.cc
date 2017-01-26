#include <stdio.h>
#include <map>
#include <string>
#include <iostream>

/// func
void keepMax(std::map<std::string,float>& theMap, const std::string key, const float value)
{
  if (theMap.find(key)==theMap.end()) //element doesn't already exist in map
    theMap[key] = value;
  else
    // any logic can go in here, eg a counter, an average, etc...
    theMap[key] = std::max(theMap[key],value);
}

/// class
class maxMap
{
public:
  typedef std::map<std::string,float>::iterator iterator;
  
  void insert(const std::pair<std::string,float> theElement)
  {
    std::string key = theElement.first;
    float value = theElement.second;
    if (m_map.find(key)==m_map.end()) //element doesn't already exist in map
      m_map[key] = value;
    else
      // any logic can go in here, eg a counter, an average, etc...
      m_map[key] = std::max(m_map[key],value);
  }
  iterator begin() { return m_map.begin(); }
  iterator end() { return m_map.end(); }
private:
  std::map<std::string,float> m_map;
};


///main
int main()
{
  FILE* ifp = fopen("02stl/cpp/particleList.txt","r");

  // Read in the data
  maxMap theParticles;
  char name[80];
  float momentum;
  while (!feof(ifp)) {
    fscanf(ifp, "%s %f", name, &momentum);
    if (!feof(ifp)) theParticles.insert( std::pair<std::string,float>(std::string(name),momentum) );
  }
  fclose(ifp);

  // Output - it's already sorted!
  maxMap::iterator iter = theParticles.begin();
  for ( ; iter!=theParticles.end(); ++iter) {
    std::cout << (*iter).first << " " << (*iter).second << std::endl;
  }      
}
