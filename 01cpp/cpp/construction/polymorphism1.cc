#include <iostream>

class Shape {
public:
  double GetArea() { return 0; }
};

class Circle : public Shape {
public:
  Circle() {};
  Circle(double r) : radius(r) {}
  double GetArea() { return 3.1415926 * radius * radius; }
private:
  double radius;
};

class Square : public Shape {
public:
  Square(double s) : side(s) {}
  double GetArea() { return side*side; }
private:
  double side;
};

int main()
{
  Shape *a = new Circle(2);
  std::cout << "a=" << a->GetArea() << std::endl;
  Square *b = new Square(2);
  std::cout << "b=" << b->GetArea() << std::endl;
  Shape c;
  return 0;
}

