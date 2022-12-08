# Pointers

You'll already have used references to refer to objects without copying them. This is very useful for passing objects to functions without copy overheads, or to functions which also change those objects in ways that we want to persist once we've left that function's scope. There are, however, times where we cannot use references and must use pointers. Some limitations of references are:
* References cannot be reassigned to refer to a new place in memory, they can only be assigned at initialisation. 
* You cannot have references of references.
* You cannot create circular references e.g. an object A which has a reference to object B, which in turn has a reference to object A. This would require reassignment of references to construct. 
* References cannot be null. 
* You cannot store references in container types like `vector`, `array`, `set`, or `map`. 

In these cases, we use *pointers*. A pointer is variable which stores an address in memory where an object's data is located (we say that it "points to" this object), or the special value `nullptr`. Pointers give us much more flexibility than references, especially when writing classes for objects that need to point to other data (either of the same class, like in graph representation where nodes point to other nodes, or of another class). In modern C++ (since C++11) we usually declare a pointer using a *smart pointer*, of which there are three different kinds: unique pointers, shared pointers, and weak pointers. 

## Optional Background: The Stack, the Heap, and Variable Scope

We will go into more detail on memory structures later on in the course when we discuss performance programming. It can however be easier to understand the usage of pointers in C++ if we understand the difference between two different kinds of memory: the _stack_ and the _heap_. 
- **Stack** memory is used to store all local variables, as well as the "call stack". The call stack contains information about the currently active functions, including the value of variables in each scope, and allows us to continue execution at the correct place in the program when we leave a function. Stack memory is fast, but limited in size. The amount of stack memory available is not known to the program at compile time, as stack memory is reserved for the program at runtime by the operating system. Using too much stack memory causes a _stack overflow_ error, which will cause your program to crash. When variables on the stack go out of scope then their destructor is called and their memory is freed. 
- **Heap** memory is somewhat slower, but can make use of as much RAM as you have available, so large datasets tend to be declared on the heap. (Heap memory is still faster than reading/writing to hard disk.) Any memory allocated on the heap _must_ be pointed to by something on the stack, otherwise it will be inaccessible to us. 

Data will end up on the stack or the heap depending on how it is declared, and the internal structure of the class itself. 
- When you declare a variable, then it is stored on the stack e.g. `int x = 5;` will store an integer on the stack. Declaring any kind of variable this way stores that object on the stack. 
- Initialising a variable by declaring a pointer using e.g. `make_unique`, `make_shared`, or `new` will allocate memory on the heap. Note however that the variable which has been declared - the pointer - is on the _stack_, and the memory that it is pointing to is on the _heap_. 
- Many objects which are not simple types will also declare memory on the heap: `vector<>` is an example of such a class. We can use the code `vector<int> v = {1,2,3,4,5};` to declare a vector `v` on the stack. The vector itself is on the stack, and is deleted when `v` goes out of scope. The data stored in the vector - in this case, five integers - is not stored on the stack, but is actually stored on the heap. The vector `v` will contain a pointer to this heap memory, and uses this to retrieve your data when you call for it. The vector class will automatically free the memory it allocates on the heap when its destructor is called, so you don't have to do that yourself. 
- Pointers do not have to point to heap memory, they can also point to stack memory if initialised with a reference to a stack variable, e.g. `int * x_ptr = &x`. In general, you should think carefully about whether you want this behaviour; as we shall see later this can lead to memory problems if not handled carefully!

## What Are Smart Pointers? 

Smart pointers are a special kind of pointer, introduced in C++11. Since then, they are typically used as the default pointers for most applications, as they automatically handle some memory management which would previously have to be done manually. The reason we have three different kinds of smart pointers is because they embody three different possible ideas about *memory ownership*. Understanding ownership is key to understanding the useage of smart pointers. 

When we talk about ownership of some memory or data, the question we are asking is what should have control over the lifetime of the data i.e. when the data should be allocated and freed. Smart pointers in C++ address three cases:
- Unique Ownership: The lifetime of the data should be determined by the lifetime of a single variable. This is essentially how we treat stack variables: when the variable goes out of scope, the destructor is called and the memory is freed. **Unique Pointers** offer a similar model for memory that is allocated on the heap. 
- Shared Ownership: The lifetime of the data is determined by multiple other variables, and the data should remain as long as one of those variables is still in scope. This is represented using **Shared Pointers**. Once all shared pointers pointing to a particular piece of data go out of scope, then the memory for that data is freed. 
- Non-Owning: Non-owning pointers should have no impact on the lifetime of the data. When the non-owning pointer goes out of scope nothing happens to the data that it was pointing to. There are represented using **Weak Pointers** or traditional raw pointers. 

## Unique Pointers `std::unique_ptr<>`

A variable should be a *unique pointer* when that is the only variable that controls whether the resource should be destroyed or not. When a unique pointer goes out of scope, the object that it pointed to is deleted and the memory freed.

```cpp
#include <memory>
#include <iostream>

int main()
{
    std::unique_ptr<int> p_int = std::make_unique<int>(17);
    std::unique_ptr<int> p_int2(new int(25))

    std::cout << *p_int << std::endl;
    std::cout << *p_int2 << std::endl;

    return 0;
}
```
- Remember that pointers actually store memory addresses, not the values of the variables that they point to. So to get the value of the variable we need to "dereference" the pointer using the `*` operator. `p_int` refers to the smart pointer, but `*p_int` gives us the value of the integer that we are pointing to. 
- We can make assignments to `*p_int` which will update the value of the integer, but doesn't change the memory location (so `*p_int` will change, but `p_int` won't). 
- Using `std::make_unique<>` and then declaring an object will create the object on the stack and then copy it to the heap. Declaring a unique pointer by passing `new` to the constructor will create the object directly on the heap, and so can be preferable in circumstances where the object being created it large and copies are computationally expensive. 

You also can't make a copy of a unique pointer, as then there would be a conflict over which one should handle the destruction of the object when it goes out of scope. This means that when we want to pass a unique pointer to a function, we cannot pass it by value, because this would involve making a copy. We can, however, pass a unique pointer by reference. 
```cpp
#include <memory>
#include <iostream>

void updatePtrValue(std::unique_ptr<int> &p)
{
    *p += 5;
}

int main()
{
    std::unique_ptr<int> p_int = std::make_unique<int>(17);
    updatePtrValue(p_int);
    std::cout << *p_int << std::endl;

    return 0;
}
```
This would usually be the preferred way to pass unique pointers to a function. If, for some reason, you don't want to pass by reference, then you will need to use `std::move`. 

Since we can't have multiple unique pointers pointing to the same data, if we want to transfer the ownership of the data to a new unique pointer, we use `std::move` as follows:
```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(5);
std::unique_ptr<int> p2 = std::move(p1);
```
- **Avoid doing this if at all possible! After this operation, `p1` will no longer point to valid memory, and will cause a segmentation fault if accessed (i.e. your program will crash).** 

This can also apply to functions if not passing a unique pointer by reference. This can lead to extremely dangerous code as we can see in this example:
```cpp
std::unique_ptr<int> updatePtrValue(std::unique_ptr<int> p)
{
    *p += 5;
    return std::move(p);
}

int main()
{
    std::unique_ptr<int> p1 = std::make_unique<int>(17);
    auto p2 = updatePtrValue2(std::move(p1));
    p2.swap(p1);
    std::cout << *p1 << std::endl;

    return 0;
}
```
- In this code, the memory is passed from `p1` to `p`, a unique pointer which is local to the function. In order to avoid the memory being deleted when we leave the function scope, we use `std::move` to move the memory from `p` to `p2`. At that point, `p2` points to our useful memory, and `p1` is dangling. We use `swap` to move the memory back to `p1`, which leaves `p2` dangling. **If we access `p2` this program will crash. Do not use `std::move` to pass data around, and only use it is you want a new variable to take control of the destruction of that data.** 
- A good example of using `std::move` would be if another object, perhaps with a broader scope than the existing data, needed to take ownership of that data as part of its class. Then if that object outlives the current scope, the data will be maintained for as long as that object lives. **In this case we still need to be careful not to access dangling pointers created by `std::move`**. 
- We can test whether a unique pointer `p` points to valid memory using `(p)` which returns `true` if it points to valid memory and `false` otherwise.
```cpp
if (p2)
{
   std::cout << *p2 << std::endl;
}
else
{
   std::cout << "empty" << std::endl;
}
```

The same considerations about using move semantics or references applies when adding unique pointers to containers such as vector. If you want the vector to be the new owner of the memory, then use `std::move`; otherwise you will have to use some kind of pointer to the unique pointer. 

## Shared Pointers `std::shared_ptr<>`

Shared pointers can be used when we might want multiple references to a single object, and we don't want that object to be destroyed until all of those references to it are gone. In other words, shared pointers model _shared ownership_, where each of the pointers has equal ownership over the object. For example, we might want to add our object to a number of lists, or have it as a member variable for other objects which may have a longer lifetime than the original pointer. In order to keep the object alive for as long as any other objects need it, we can declare it as a shared pointer:

```cpp
#include <memory> 

int main()
{
    std::shared_ptr<int> sp_x = std::make_shared<int>(12);
    std::shared_ptr<int> sp_x2(sp_x); 
}
```
- `sp_x` and `sp_x2` both point to the same location in memory, which holds an int of value `12`. 
- We can change the value in memory by altering `*sp_x` or `*sp_x2`; note that since they point to the same memory, changing one affects the other. 
- The memory will not be freed until _both_ `sp_x` and `sp_x2` go out of scope. 

Shared pointers keep a count of the number of shared pointers which point at the same piece of data. When a new shared pointer is created to point to the data, or an existing shared pointer is changed to point to the data, then the count is increased. If a shared pointer goes out of scope or is redirected towards other data, then the count is decreased. If the count reaches zero, then that data is deleted and the memory freed. 

Because shared pointers _share_ ownership of an object, they can be copied, and they can be passed by value into functions or into containers such as `vector`. The move operations available to unique pointers are also available to shared pointers, if you want to reuse a shared pointer to point to something else. 

**N.B.** Many programmers default to using shared pointers because they have fewer restrictions; try to only use them when they properly model the ownership of the data in question. Shared pointers incur overheads and can cause memory issues of their own if not properly managed, as we shall see below.

### Circular references 

When we're using shared pointers, we need to be very wary of circular references. Consider a simple class like this:
```cpp
class Person
{
    public:
    Person(std::string name)
    {
        name = name;
        std::cout << name << " created" << std::endl;
    }
    ~Person()
    {
        std::cout << name << " destroyed" << std::endl;
    }

    void SetFriend(std::shared_ptr<Person> &otherPerson)
    {
        bestFriend = otherPerson;
    }

    private:
    std::shared_ptr<Person> bestFriend;
    std::string name;
};
```
In this class, a person can have one best friend, and they are kept track of using a shared pointer. This is because a person doesn't have ownership of their friend - if one of them goes out of scope and is deleted, it shouldn't cause the other one to be deleted! 

If we create a pair of `Person` objects, and use the `SetFriend` method to make each `Person` point to the other as follows:
```cpp
int main()
{
    std::shared_ptr<Person> Alice(new Person("Alice"));
    std::shared_ptr<Person> Bob(new Person("Bob"));

    Alice->SetFriend(Bob);
    Bob->SetFriend(Alice);

    return 0;
}
```
we will end up creating a circular reference with shared pointers! Alice now has a shared pointer to Bob, and Bob has a shared pointer to Alice. We can see the problem with this by looking at the output of this program:
```
Alice created
Bob created
```
We can see that neither destructor is ever called, even though both pointers have gone out of scope.
When Alice, for example, goes out of scope then Alice's data stays live because Bob has a reference to her. This means that Alice's shared pointer pointing to Bob still exists. Then if Bob goes out of scope next, then Bob can't delete his own data because Alice's shared pointer to Bob hasn't been deleted. As a result they both still have a reference count of 1, and cannot be deleted. This causes a memory leak, since their data will remain allocated until the program is terminated and all memory is freed. This kind of behaviour should always be avoided. 

- This is an example of using shared pointers in a situation where it does not correctly model the ownership of the data. In this model a `Person` shouldn't have control over the lifetime of another, since different `Person` objects should be allowed to be created or destroyed independently. One `Person` is not a part of, or owned, by another. 
- In order to model this concept properly we need to use *non-owning* pointers, that allow objects to point to one another without influencing their lifetime. 

## Weak Pointers `std::weak_ptr<>`

Weak pointers are a special kind of smart pointer which can only point to memory owned by a shared pointer. You cannot use weak pointers to initialise new objects in memory, point to memory owned by unique pointers, or point to references of ordinary objects. Weak pointers do not contribute to the pointer count for the shared pointer, so they do not impact the lifetime of the object that they are pointing to. They can be used therefore to break the circular reference problem with shared pointers and model situations where there is no ownership relation. 

Because weak pointers do not own the memory to which they point, that memory can be freed (the object deleted) before the weak pointer is out of scope, leaving the weak pointer to point at invalid memory. You can check whether a weak pointer points to valid memory using the same method as unique or shared pointers: if we have a weak pointer `wpt` then the expression `(wpt)` will evaluate to `true` if `wpt` is pointing to an object which has not been destroyed, and `false` if the object has been deleted. This is especialy important for weak pointers as this can happen even if the programmer does not invoke any `move` or `swap` operations. Although it can slow down execution, it is a good idea to check that the existence of objects pointed to by weak pointers before trying to access them. You can also use `wpt.expired()` which will return `true` if the memory is deleted and `false` otherwise (the opposite of `(wpt)`). 

Accessing weak pointers is also different to accessing other kinds of pointers because they cannot be dereferenced directly.
That means if we have a weak pointer `wpt` we can't get the value using `*wpt` or call a member function using `wpt->function()`. Instead, we must create a new shared pointer to that memory using `spt_new = wpt.lock()`, and then access the data through `spt_new`. This also creates additional overheads for accessing weak pointers. 

If we modify our above example to use a `weak_ptr` for `bestFriend` instead of a `shared_ptr`:
```cpp
class Person
{
    public:
    Person(std::string name)
    {
        name = name;
        std::cout << name << " created" << std::endl;
    }
    ~Person()
    {
        std::cout << name << " destroyed" << std::endl;
    }

    void SetFriend(std::shared_ptr<Person> &otherPerson)
    {
        bestFriend = otherPerson;
    }

    private:
    std::weak_ptr<Person> bestFriend;
    std::string name;
};

int main()
{
    std::shared_ptr<Person> Alice(new Person("Alice"));
    std::shared_ptr<Person> Bob(new Person("Bob"));

    Alice->SetFriend(Bob);
    Bob->SetFriend(Alice);

    return 0;
}
```
then the circular dependency is broken and we get the output:
```
Alice created
Bob created
Bob Destroyed
Alice Destroyed
```
showing that both destructors are now called and the memory freed. 

## Memory Ownership

Smart pointers are designed to model the ownership of memory allocated on the heap. **You should never create smart pointers which point to variables already declared on the stack.** Objects declared on the stack are destroyed when the original variable goes out of scope, and it is not possible to transfer this ownership. Therefore, any smart pointer pointing to a stack object *cannot* properly model the ownership of the data. If the original variable goes out of scope first, then your pointers will be left dangling. If you pointer goes out of scope first, then your program will crash even if the original variable isn't accessed again, because the smart pointer will not have the right to delete that data. If you need to point to stack data, you will need to use a *raw pointer*, which does not embody any ownership.  

## Raw Pointers

You may have come across *raw pointers* before when using C++. These have existed in C++ for longer (indeed, they pre-date C++), and work slightly differently to smart pointers. The most important difference is that **raw pointers don't do any automatic memory management**. This means that if you use a raw pointer to assign some memory, and you do not manually free that memory before the pointer goes out of scope, that memory will become inaccessible and will not be freed until your program terminates. This is called a memory leak, and can be extremely important when running large programs, where you risk running out of memory and therefore prematurely aborting your program execution. 

**Avoid using raw pointers for objects whose memory is not owned by some other resource** (e.g. a stack variable, a container, or a smart pointer). Raw pointers can be used for "non-owning pointers", which have no impact on the lifetime of the object that they point to, **as long as you can be sure that they won't point to invalid memory**. This can be useful for graph like classes where objects can reference one another (including circular references) but do not influence the lifetime of each other, or for referencing objects in container classes like `vector`. Bear in mind however, that we can't check if a raw pointer points to valid memory of not, so if the object that the raw pointer points to is deleted before the raw pointer, then accessing that pointer will cause undefined behaviour.
- Okay for non-owning pointers 
- Raw pointers should be used when you need to point to an existing stack variable. **Never point to an existing stack variable using smart pointers as their memory management will conflict.**
- Faster memory access than weak pointers, although unique/shared pointers can match speed of raw pointers if compiled with optimisation.  
- Lower memory management overheads. 
- Must manually use `new` and `delete` to manage memory allocation/deallocation: this is a frequent source of bugs. 
- Can't check if memory pointing to has already been freed; if you need to do this then you should use a combination of shared and weak pointers. 
- **Only use when you know they are safe and have specific performance considerations**

## Using `const` with pointers 

Remember that we declare variables to be read only using the `const` keyword, for example
```
int const x = 5;
```
makes `x` a read only variable with the value `5`. We can retrieve and use the value of `x`, but we can't update it. We can also write this
```
const int x = 5;
```
which is exactly equivalent, but the `int const` form is preferred because it is more consistent with the pointer notation which we will look at next. 

Using `const` with pointers allows us to declare one of two things (or both):
1. The pointer points to a `const` type: we declare the data pointed to constant, and so this pointer cannot be used to update the value of held in the memory location to which it points. In other words, the memory pointed to is declared read-only, and we can deference the pointer to retrieve the data at that location, but we can't update it. We can however change the memory address that the pointer points to, since the pointer itself is not constant (remember the pointer is actually a variable storing a memory address). 
    - To do this with a smart pointer we need to place the `const` in the angle brackets, e.g. `shared_ptr<const int> readOnlySPtr` or `shared_ptr<int const> readOnlySPtr` which declares a shared pointer to a constant int. The `const` keywork here applies to the type of the data, `int`, so it is the data pointer to, not the pointer itself, which is being declared const. 
    - To do this with a raw pointer use the `const` keyword _before_ the `*` operator, e.g. `int const * readOnlyPtr` or `const int * readOnlyPtr`. This declares a (raw) pointer to a constant int. 
    - A pointer to const data only prohibits the value in memory being changed _through that pointer_, but if the value can be changed another way (e.g. it is a stack variable or there is another pointer to it) then it could still be changed. 

2. The pointer itself is const: the memory location pointed to is a constant. In this case, the value held in the memory can change, but the pointer must always point to the same place and we can't redirect the pointer to look at another place in memory. 
    - We declare a smart pointer like this by placing the `const` keyword outside of the angle brackets, e.g. `shared_ptr<int> const fixedAddressSPtr` or `const shared_ptr<int> fixedAddressSPtr`. The `const` keyword is applied to the type `shared_ptr<int>` so it is the pointer itself, not the data it points to, which is constant. 
    - We declare a raw pointer in this way by placing the `const` keyword _after_ the `*` operator, e.g. `int * const fixedAddressPtr`. In this case we are applying the `const` to the type `int *` i.e. the pointer type, so the pointer itself is constant. 

3. We can combine these to declare a constant pointer to constant data by using a `const` keyword _before and after_ the `*` operator:
    - For smart pointers we can write `shared_ptr<int const> const readOnlyFixedPtr`.
    - For raw pointers we can write `int const * const readOnlyFixedPtr` or `const int * const readOnlyFixedPtr`.

**In general, `const` binds with the type to its left, unless it is in the leftmost position in which case it will bind with the first type to the right.** Therefore `int * const p` declares `p` to be a constant (`const`) pointer (`*`) to an integer (`int`), and `int const * p` and `const int * p` both declare `p` to be a pointer to a constant integer.

Use of the `const` keyword is especially important when using pointers to pass arguments to functions. As we have seen in our discussion of pass by reference, passing a memory location to a function is efficient (as it prevents copying of data) but comes with risks of altering data which persist outside of the function scope. Just as in the case of passing by reference, declare the pointers read only using the `const` keyword in your function signature whenever you can (i.e. whenever the data is not intended to be updated in place). We can pass a non-`const` pointer into a function which accepts `const` pointers, but we cannot pass a `const` pointer into a function which accepts non-`const` pointers as the compiler can't be sure that the function won't try to modify the pointer, or the data it points to, in ways which are not allowed. 
```cpp
int addWithConstPointers(int const * const a, int const * const b)
{
    return (*a) + (*b);
}
```

## Using libraries which use raw pointers

Some commonly used external libraries like LAPACK or GNU Science Library, which are C compatible, will require you to pass raw pointers to functions. In this case, you do not have to forego using smart pointers in your own code. A smart pointer is a wrapper for a raw pointer, and the raw pointer can be retrieved using `.get()` on a smart pointer. You can then pass this raw pointer to the library function. 

These functions will usually use C-style arrays, which are pointers declared and used in a function in the following way:
```cpp
// N by N float matrix
int N = 1000;
float *myFloatMatrix = new float[N * N]
int res = myMethod(myFloatMatrix);
```

To use a smart point we could write:
```cpp
int N = 1000;
std::unique_ptr<float[]> myFloatMatrix = std::make_unique<float[]>(N*N);
int res = myMethod(myFloatMatrix.get());
```
In general we will not focus on libraries of this kind, but it is good to be aware of them as you may come across them in the future when working with academic software. 

## Smart Pointers, Performance, and Optimisation

Smart pointers are more complex than raw pointers, and can incur some overheads. Shared pointers require some additional memory in order to perform the reference counting. The compiler can automatically optimise away almost all of the time overhead associated with shared or unique pointers, but without optimisation they can be significantly slower. We'll discuss how to perform this kind of optimisation at compile time later on when we discuss programming for high performance.

Accessing weak pointers will remain slow even with optimisation turned on; this is because of the `.lock()` operation, which has to make a copy of an existing shared pointer before it can be dereferenced. If accessing pointers is a bottleneck in your program (for example a graph traversal where nodes are linked by pointers), then you may want to consider replacing the weak (and therefore non-owning) pointers with raw pointers if you can ensure that they won't attempt to dereference invalid memory. 

## Memory Problems

All kinds of pointers can cause some memory problems through improper handling. Whenever you use pointers, make sure that you're careful, understand when your memory is being allocated and deallocated, test your functions thoroughly for memory access issues and use a profiler to check for memory leaks. We'll discuss profiling memory usage using `valgrind` in a later lecture. 