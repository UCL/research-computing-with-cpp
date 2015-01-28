#include <iostream>
class A {
  public:
  protected:
    A() { std::cout << "A constructor" << std::endl;}
    virtual ~A() { std::cout << "A destructor" << std::endl;}
  private:
    A(const A &); //purposely not implemented
    void operator=(const A &);  //purposely not implemented
};
class B : public A {
  public:
    B() { std::cout << "B constructor" << std::endl;}
    virtual ~B() { std::cout << "B destructor" << std::endl;}
  protected:
  private:
    B(const B &); //purposely not implemented
    void operator=(const B &);  //purposely not implemented
};
int main()
{
  //A a;
  B b;
  return 0;
}