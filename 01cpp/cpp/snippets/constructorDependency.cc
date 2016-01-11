#include <memory>

class Bar {
};

class Foo {
public:
  Foo()
  {
    m_Bar.reset(new Bar());
  }

private:
  std::unique_ptr<Bar> m_Bar;
};

int main()
{
  Foo a;
}

