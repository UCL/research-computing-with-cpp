#include <memory>

class Bar {
};

class Foo {
public:
  Foo(Bar* b)
  : m_Bar(b)
  {
  }

private:
  std::unique_ptr<Bar> m_Bar;
};

int main()
{
  Bar b;
  Foo a(&b);
}

