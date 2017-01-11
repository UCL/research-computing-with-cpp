#include <memory>

class Bar {
};

class Foo {
public:
  Foo()
  {
  }
  void SetBar(Bar *b) { m_Bar.reset(b); }

private:
  std::unique_ptr<Bar> m_Bar;
};

int main()
{
  Bar b;
  Foo a();
  a.SetBar(&b);
}

