class MyFilter : // Other stuff
{
public:
  typedef MyFilter Self
  typedef SmartPointer<Self> Pointer;
  itkNewMacro(Self);
protected:
  MyFilter();
  virtual ~MyFilter();
};

double someFunction(MyFilter::Pointer p)
{
  // stuff
}

int main()
{
  MyFilter::Pointer p = itk::MyFilter::New();
}

