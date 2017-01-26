/// viter
std::vector<int> myVector = {1,2,3,4};
for (int i=0; i<myVector.size(); ++i) {
    std::cout << myVector[i] << std::endl;
}


///liter
#include <iterator>

std::list<int> mylist = {1,2,3,4};
std::list<int>::iterator it=mylist.begin()
for ( ; it!=mylist.end(); ++it) {
    std::cout << *it << std::endl;
}

    
/// pairs
#include <utility>      // std::pair

int main () {
  std::pair <int,int> foo;                                     // default constructor
  std::pair <std::string,double> product1;                     // default constructor
  std::pair <std::string,double> product2("tomatoes",2.30);   // value init

  foo = std::make_pair (10,20);
  product1 = std::make_pair(std::string("lightbulbs"),0.99);   // using make_pair

  product2.first = "shoes";                  // the type of first is string
  product2.second = 39.90;                   // the type of second is double

  std::cout << "foo: " << foo.first << ", " << foo.second << '\n';
  std::cout << "bar: " << bar.first << ", " << bar.second << '\n';

  return 0;
}


/// tuples
#include <tuple>        // std::tuple, std::make_tuple, std::get

int main()
{
  std::tuple<int,char> one;                          // default
  std::tuple<int,char> two(10,'a');                  // initialization

  auto first = std::make_tuple (10,'a');             // tuple < int, char >
  const int a = 0; int b[3];                         // decayed types:
  auto second = std::make_tuple (a,b);               // tuple < int, int* >

  std::cout << "two contains: " << std::get<0>(two);
  std::cout << " and " << std::get<1>(two);
  std::cout << std::endl;

  return 0;
}
