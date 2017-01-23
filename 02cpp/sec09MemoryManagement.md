---
title: Smart Pointers
---

{% idio cpp %}

## Smart Pointers

### Use of Raw Pointers

* Given a pointer passed to a function

```
   void DoSomethingClever(int *a) 
   {
     // write some code
   }
```

* How do we use the pointer? 
* What problems are there?


### Problems with Raw Pointers

* From ["Effective Modern C++", Meyers, p117](https://www.amazon.co.uk/Effective-Modern-Specific-Ways-Improve/dp/1491903996/ref=sr_1_1?ie=UTF8&qid=1484571499&sr=8-1&keywords=Effective+Modern+C%2B%2B).
    * If you are done, do you destroy it?
    * How to destroy it? Call ```delete``` or some method first: ```a->Shutdown();```
    * Single object or array?
    * ```delete``` or ```delete[]```?
    * How to ensure the whole system only deletes it once?
    * Is it dangling, if I don't delete it?


### Use Smart Pointers

* ```new/delete``` on raw pointers not good enough
* So, use Smart Pointers
    * automatically delete pointed to object
    * explicit control over sharing
    * i.e. smarter
* Smart Pointers model the "ownership"


### Further Reading

* Notes here are based on these:
    * [David Kieras online paper](DavidK)
    * ["Effective Modern C++", Meyers, ch4](Meyers14)


### Standard Library Smart Pointers

* Here we teach Standard Library
    * [std::unique_ptr](http://en.cppreference.com/w/cpp/memory/unique_ptr) - models *has-a* but also unique ownership
    * [std::shared_ptr](http://en.cppreference.com/w/cpp/memory/shared_ptr) - models *has-a* but shared ownership
    * [std::weak_ptr](http://en.cppreference.com/w/cpp/memory/weak_ptr) - temporary reference, breaks circular references


### Stack Allocated - No Leak.

* To recap:

{% code memory/fractionOnStack.cc %}

* Gives:

{% code memory/fractionOnStack.out %}

* So stack allocated objects are deleted, when stack unwinds.


### Heap Allocated - Leak.

* To recap:

{% code memory/fractionOnHeap.cc %}

* Gives:

{% code memory/fractionOnHeap.out %}

* So heap allocated objects are not deleted.
* Its the pointer (stack allocated) that's deleted.


### Unique Ptr - Unique Ownership

* So:

{% code memory/fractionOnHeapUniquePtr.cc %}

* Gives:

{% code memory/fractionOnHeapUniquePtr.out %}

* And object is deleted.
* Is that it?


### Unique Ptr - Move?

* Does move work?

{% code memory/fractionUniquePtrMove.cc %}

* Gives:

{% code memory/fractionUniquePtrMove.out %}

* We see that API makes difficult to use incorrectly.


### Unique Ptr - Usage 1

* Forces you to think about ownership 
    * No copy constructor
    * No assignment
* Consequently
    * Can't pass pointer by value
    * Use move semantics for placing in containers
    

### Unique Ptr - Usage 2

* Put raw pointer STRAIGHT into unique_ptr
* see ```std::make_unique``` in C++14.

{% code memory/fractionOnHeapUniquePtr.cc %}


### Shared Ptr - Shared Ownership

* Many pointers pointing to same object
* Object only deleted if no pointers refer to it
* Achieved via reference counting


### Shared Ptr Control Block

* Won't go to too many details:
<img src="https://www.safaribooksonline.com/library/view/effective-modern-c/9781491908419/assets/emcp_04in02.png" alt="Control Block">

* From ["Effective Modern C++", Meyers, p140](Meyers14)


### Shared Ptr - Usage 1

* Place raw pointer straight into shared_ptr
* Pass to functions, reference or by value
* Copy/Move constructors and assignment all implemented


### Shared Ptr - Usage 2

{% code memory/fractionOnHeapSharedPtr.cc %}


### Shared Ptr - Usage 3

* Watch out for exceptions.
* ["Effective Modern C++", Meyers, p140](Meyers14)

{% code memory/fractionExceptionProblem.cc %}


### Shared Ptr - Usage 4

* Prefer ```std::make_shared```
* Exception safe

{% code memory/fractionExceptionMakeShared.cc %}


### Weak Ptr - Why?

* Like a shared pointer, but doesn't actually own anything
* Use for example:
    * Caches
    * Break circular pointers
* Limited API
* Not terribly common as most code ends up as hierarchies


### Weak Ptr - Example

* See [David Kieras online paper](DavidK)
        
{% code memory/fractionOnHeapWeakPtr.cc %}
        
### Final Advice

* Benefits of immediate, fine-grained, garbage collection
* Just ask [Scott Meyers!](Meyers14)
    * Use ```unique_ptr``` for unique ownership
    * Easy to convert ```unique_ptr``` to ```shared_ptr```
    * But not the reverse
    * Use ```shared_ptr``` for shared resource management
    * Avoid raw ```new``` - use ```make_shared```, ```make_unique```
    * Use ```weak_ptr``` for pointers that can dangle (cache etc)


### Comment on Boost

* Boost has become a sandbox for standard C++
* Boost features become part of standard C++, (different name space)
* So if you are forced to use old compiler
    * You could use boost - lecture 5.


### Intrusive Vs Non-Intrusive

* Intrusive - Base class maintains a reference count eg. [ITK](http://www.itk.org)
* Non-intrusive
    * ```std::unique_ptr```
    * ```std::shared_ptr```
    * ```std::weak_ptr```
    * works for any class
    

### ITK (intrusive) Smart Pointers

{% code snippets/itkSmartPointer.cc %}


### Conclusion for Smart Pointers

* Default to standard library, check compiler
* Lots of other Smart Pointers
    * [Boost](http://www.boost.org) (use STL).
    * [ITK](http://www.itk.org)
    * [VTK](http://www.vtk.org/Wiki/VTK/Tutorials/SmartPointers)
    * [Qt Smart Pointers](https://wiki.qt.io/Smart_Pointers)
* Don't be tempted to write your own
* Always read the manual
* Always consistently use it


{% endidio %}

Meyers14: https://www.amazon.co.uk/Effective-Modern-Specific-Ways-Improve/dp/1491903996/ref=sr_1_1?ie=UTF8&qid=1484571499&sr=8-1&keywords=Effective+Modern+C%2B%2B
DavidK: http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf