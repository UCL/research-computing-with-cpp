#include <memory>

class Bar {
};

class Foo {
public:
  Foo()
  {
    m_Bar = new Bar();
  }

private:
  Bar* m_Bar;
};

int main()
{
  Foo a;
}

