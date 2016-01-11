#include <iostream>
#include <memory>
 
struct Foo
{
    Foo()      { std::cout << "Foo::Foo\n";  }
    ~Foo()     { std::cout << "Foo::~Foo\n"; }
    void bar() { std::cout << "Foo::bar\n";  }
};
 
void f(const Foo &) { std::cout << "f(const Foo&)\n"; }

int main()
{
    std::unique_ptr<Foo> p1(new Foo);  // p1 owns Foo
    if (p1) p1->bar();
 
    // Foo instance is destroyed when p1 goes out of scope
}

