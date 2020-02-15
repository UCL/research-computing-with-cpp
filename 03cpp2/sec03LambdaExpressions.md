---
title: Lambda expressions 
---

{% idio cpp %}

## Lambda expressions  

### Game changer for C++ 

* Lambda expressions allow you to create an unnamed function object within code
   * Expression can be defined as they are passed to a function as an argument
   * Useful for varions STL `_if` and comparison functions: `std::find_if`, `std::remove_if`, `std::sort`, `std::lower_bound` etc
   * Can be used for a one off call for a context specific function

* From “Effective Modern C++”, Meyers, p215
   * A game changer for C++ despite bringing no new expressive power to the language
   * Everything you can do with a lambda could be done by hand with a bit more typing
   * But the impact on day to day C++ software development is enormous  
   * Allow expressions to be defined as they are being passed as an argument to a function

``` cpp
   std::find_if(container.begin(), container.end(),
                [](int val) { return 0 < val && val < 10; });
``` 

### Basic syntax

* `[ captures ] { body };`
   * `captures` is comma separated list of variable from enclosing scope that the lambda can use
   * `body` is where the function is defined
   * `[x,y] { return x + y; }` 
   * Captures can be by value `[x]` or by reference `[&x]`
   * `[=]` and `[&] are default capture modes for all variables in enclosing scope -> discouraged as can lead to dangling references and are not thread safe ("Effective Modern C++", Meyers, p216) 

* `[ captures ] ( params ) { body };`
   * `params` list of params as with named functions except cannot have default value
   * `[] (int x, int y) { return x*x + y*y; }(1,2)` would return `5` 

* `[ captures ] ( params ) -> ret { body };`
   * `ret` is the return type, if not specified inferred from the return statement in the function
   * `[] () -> float { return "a"; }` would give `error: cannot convert const char* to float in return`
  
* It is possible to copy and reuse lambdas

```
  auto c1 = [](int y) { return y*y; };
  auto c2 = c1; // c2 is a copy of c1
  cout << "c1(2) = " << c1(2) << endl;
  cout << "c2(4) = " << c2(4) << endl;
```
* Gives

``` cpp
c1(2) = 4
c2(4) = 16
```
 
### Example use

``` cpp
  std::vector<int> v { 1,2,3,4,5,6,7,8,9,10 };
  int neven = std::count_if(v.begin(), v.end(), [](int i){ return i % 2 == 0; });
```

### Homework 15

* Create your own lambda expressions for each of the three basic syntax examples given above
* Try to change a param from within, can you see a different behaviour if passed by reference or by value?
* Use `std::count_if` with an appropriate lambda expression to count the number of values in a `vector<int>` that are divisable by 2 or 3   

### Homework 16

* Create a simple `Student` class that has public member variables storing `string firstname`, `string secondname` and `int age` 
* Create a vector and fill it with varios instances of the `Student` class 
* Create a sorting class `StudentSort` and that has a method `vector<Student> SortByAge(vector<Student> vs);` that returns a `vector<Student>` that has been sorted by age
   * Use a `std::sort` and a lambda expression for this
   * Add a `bool` switch to `SortByAge(vector<Student> vs, bool reverse = false)` that reverses the sort

{% endidio %}

