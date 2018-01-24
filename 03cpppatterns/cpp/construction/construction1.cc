#include <iostream>
#include <string>

class Base {
public:
  Base(std::string n) 
  : name(n) 
  { 
    std::cout << "1:" << name << std::endl; 
  }
  virtual ~Base() 
  { 
    std::cout << "2:" << name << std::endl; 
  }
  std::string GetName() const 
  { 
    return name; 
  }
private:
  std::string name;
}; 

class Derived : public Base {
public:
  Derived(std::string n) 
  : Base(n) 
  { 
    std::cout << "3:" << this->GetName() << std::endl; 
  }
  virtual ~Derived() 
  { 
    std::cout << "4:" << this->GetName() << std::endl; 
  }
};

int main() {
  Base *a;
  Derived b("b");
  a = new Base("a");  
  return 0;
}
