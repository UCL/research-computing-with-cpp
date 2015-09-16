#ifndef explicitInstantiation_h
#define explicitInstantiation_h
#include <iostream>
template<typename T>
void f(T s)
{
    std::cout << typeid(T).name() << " " << s << '\n';
}
#endif
