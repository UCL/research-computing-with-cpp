---
title: Other STL stuff
---

## Other STL stuff

### Contents

* I/O with streams
* Function objects
* Algorithms


### Streams

* [iostream](http://www.cplusplus.com/reference/iostream/) for `std::cout`, `std::cerr` and `std::cin`
* [fstream](http://www.cplusplus.com/reference/fstream/) for file I/O
* [sstream](http://www.cplusplus.com/reference/sstream/) for string manipulation

```cpp
std::ifstream myfile(filename,std::ifstream::in);
if (!myfile.good()) {
   stringstream mess;
   mess << "Cannot open file " << filename << " . It probably doesn't exist." << endl;
   throw runtime_error(mess.str());
}

myfile.seekg(0, std::ios::end);
m_size = myfile.tellg();
m_buffer = new char[m_size];
myfile.seekg(0, std::ios::beg);
myfile.read(m_buffer,m_size);
myfile.close();
```


### Function objects

* Remember `std::set<std::string,compMass> theParticles;` ?
* There are several ways to define function-type stuff in c++, and `std::function` wraps them all
    * functions and function pointers
    * functors (i.e. an object with an `operator()` member function)
    * lambdas (i.e. nameless inline functions)
* Such functions can be used as comparators in STL containers or algorithms, among other things.


### Function objects

```cpp
#include <functional>   // std::function, std::negate

// a function:
int half(int x) {return x/2;}

// a function object class:
struct third_t {
  int operator()(int x) {return x/3;}
};

// a class with data members:
struct MyValue {
  int value;
  int fifth() {return value/5;}
};

int main () {
  std::function<int(int)> fn1 = half;                    // function
  std::function<int(int)> fn2 = &half;                   // function pointer
  std::function<int(int)> fn3 = third_t();               // function object
  std::function<int(int)> fn4 = [](int x){return x/4;};  // lambda expression
  std::function<int(int)> fn5 = std::negate<int>();      // standard function object

  std::cout << "fn1(60): " << fn1(60) << '\n';
  std::cout << "fn2(60): " << fn2(60) << '\n';
  std::cout << "fn3(60): " << fn3(60) << '\n';
  std::cout << "fn4(60): " << fn4(60) << '\n';
  std::cout << "fn5(60): " << fn5(60) << '\n';

  // stuff with members:
  std::function<int(MyValue&)> value = &MyValue::value;  // pointer to data member
  std::function<int(MyValue&)> fifth = &MyValue::fifth;  // pointer to member function

  MyValue sixty {60};
  std::cout << "value(sixty): " << value(sixty) << '\n';
  std::cout << "fifth(sixty): " << fifth(sixty) << '\n';
}
```


### Function objects

Output:
```
fn1(60): 30
fn2(60): 30
fn3(60): 20
fn4(60): 15
fn5(60): -60
value(sixty): 60
fifth(sixty): 12
```


### Algorithms

<http://www.cplusplus.com/reference/algorithm/>

* A collection of functions especially designed to be used on ranges of elements.
* Don't start coding any task, before checking if it's already there!

```cpp
std::sort(RandomAccessIter first, RandomAccessIter last, Compare comp)

std::transform(InIter first, InIter last, OutIter res, UnaryOp op)
std::transform(InIter1 first1, InIter1 last1, InIter2 first2, OutIter res, BinaryOp op)

std::for_each (InIter first, InIter last, Function fn)

std::max_element(ForwardIter first, ForwardIter last, Compare comp)
std::min_element(ForwardIter first, ForwardIter last, Compare comp)
std::minmax_element(ForwardIter first, ForwardIter last, Compare comp)
```
