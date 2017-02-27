#ifndef explicitInstantiation_h
#define explicitInstantiation_h
#include <iostream>
#include <typeinfo>
template <typename T> void f(T s) { std::cout << typeid(T).name() << " " << s << '\n'; }
#endif
