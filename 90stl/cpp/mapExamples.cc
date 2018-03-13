#include <map>
#include <string>
#include <iostream>

int main()
{
/// fill
  std::map<std::string,int> myMap;
  myMap["brocolli"] = 2;
  myMap["garlic"] = 1;
  myMap["brocolli"] = 1; // returns reference to element => element is updated
  myMap.insert( std::pair<std::string,int>("bread",4) );
  myMap.insert( std::pair<std::string,int>("brocolli",3) ); // returns iterator
                               // to existing element => element is not updated
  
  typedef std::multimap<std::string,int> MMapType;
  MMapType myMMap;                                // there's no [] for multimap
  myMMap.insert( std::pair<std::string,int>("brocolli",2) );
  myMMap.insert( std::pair<std::string,int>("bread",4) );
  myMMap.insert( std::pair<std::string,int>("brocolli",3) );

/// read
  std::cout << "myMap[brocolli] = " << myMap["brocolli"] << "\n";
  std::cout << "myMap.find(bread) = " << myMap.find("bread")->second << "\n";
  for (std::map<std::string,int>::iterator it=myMap.begin(); it!=myMap.end(); ++it)
    std::cout << it->first << " " << it->second << "\n";

  std::cout << "myMMap.count(brocolli) = " << myMMap.count("brocolli") << "\n";
  std::pair < MMapType::iterator,MMapType::iterator> range = myMMap.equal_range("brocolli");
  for (MMapType::iterator it=range.first; it!=range.second; ++it)
    std::cout << it->first << " " << it->second << "\n";
  for (MMapType::iterator it=myMMap.begin(); it!=myMMap.end(); ++it)
    std::cout << it->first << " " << it->second << "\n";
/// theend
}
