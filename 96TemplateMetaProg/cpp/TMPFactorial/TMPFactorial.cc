#include <iostream>
using namespace std;

template <int n>
struct factorial {
	enum { value = n * factorial<n - 1>::value };
};

template <>
struct factorial<0> {
	enum { value = 1 };
};

int main () {
  std::cout << factorial<0>::value << std::endl;
  std::cout << factorial<8>::value << std::endl;
}
