---
title: STL Containers
---

{% idio cpp %}

## STL Containers

### Two general types:

* *Sequences*:
Elements are ordered in a linear sequence. Individual elements are accessed by their position in this sequence/index.
eg. `myVector[3]`, `myList.front()`, etc.
  
* *Associative*:
Elements are referenced by their key and not by their position in the container.
eg. `myMap['key1']`


### Sequences

Properties:

* size: fixed/dynamic
* access: random/sequential
* underlying memory structure: contiguous/not
    - random access in non-contiguous memory is tricky
    - affects how efficiently inserting/removing of elements can be done
    - affects if pointer arithmetic can be done
* optimised insert/remove operations


### Sequences

| Name   | size | access | memory | efficient insert/remove |
| ------ |:----:|:------:|:------:|:-----------------------:|
| array  |fixed | random |contiguous| - |
| **vector** |**dynamic**|**random** |**contiguous**| **at end only** |
| deque  |dynamic|random |non-contiguous| both ends |
| list|dynamic|sequential|non-contiguous| anywhere |
|forward_list|dynamic|sequential, only forward|non-contiguous| anywhere |


### More on vector

It's dynamic, so you can add/erase elements:

```cpp
std::vector<int> myVec;
for (int i=0;i<10;++i) {
myVec.push_back(i);
}
myVec.insert(myVec.begin(),-1);
myVec.erase(1);
```

`myVec = -1 1 2 3 4 5 6 7 8 9`

... or manipulate its size:

```cpp
std::vector<int> myVec(10);
std::cout << "Vector size before = " << myVec.size();
myVec.resize(5);
std::cout << " after = " << myVec.size() << "\n";
```

`Vector size before = 10 after = 5`


### Note on C++11

* For all containers, `emplace(const_iterator position, Args&&... args)` is preferable to `insert(const_iterator position, const value_type& val)`, as it doesn't create any copies of the object you add to the container.
* Cases you might prefer something other than `emplace`:
    * backward compatibility
    * `insert` has more constructors
    * at least `emplace_back` might not work as expected in some implementations


### Exercise

Think of cases where you'd use a specific container


### Associative containers

Properties:

* key: is it separate from value?
    * maps: key-value pair
    * sets: value is the key (and thus it's const!)
* ordering. Affects performance:
    * unordered containers fastest to fill and access by key
    * ordered containers fastest to iterate through, and they're already ordered :o)
* unique values?


### Associative containers

|             | ordered | unordered |
|-------------|:-------:|:---------:|
| **unique**  | map     | unordered_map|
| **non-unique**| multimap |unordered_multimap|

|             | ordered | unordered |
|-------------|:-------:|:---------:|
| **unique**  | set     | unordered_set|
| **non-unique**| multiset |unordered_multiset|


### More on maps

Fill them with

{% fragment fill, mapExamples.cc %}


### More on maps

Access elements with

{% fragment read, mapExamples.cc %}

{% code mapExamples.out  %}

### Example

Read in file with unknown number of particle-momentum pairs.

{% code particleList.txt %}
* Then print out
    1. list of all particle-momentum pairs in alphabetical order
        * how about in ascending momentum order?
    2. list of types of particles in the file in alphabetical order
        * how about in mass order?
    3. list of particle-max momentum pairs


### Task 1

{% code particlesMMap.cc %}


### Task 2

* Hint1: use a set
* Hint2: write a custom comparator


### Task 2

* Hint1: use a set
* Hint2: write a custom comparator

{% fragment comparator, particlesSet.cc %}
then use it in the set constructor:

`std::set<std::string,compMass> theParticles;`


### Task 3

Hint: extend the map class. - How?

* inherit from std::map - **NO**!
    * STL containers are *not* designed to be polymorphic, as they don't have virtual destructors (*Meyers 1:7*)
* composition - write a class that contains either an std::map object or smart pointer to it.
* free functions with STL containers/iterators as arguments


### Task 3 - with function

{% fragment func, particlesMap.cc %}

Then use it in the loop when filling the container:

`if (!feof(ifp)) keepMax( theParticles,name,momentum );`


### Task 3 - with composition

{% fragment class, particlesMap.cc %}


### Task 3 - with composition

Then use it in main:
{% fragment main, particlesMap.cc %}


### Accessing containers: iterators

For random-access containers you can do

{% fragment viter, snippets.cc %}

But for sequential-access ones, you can only do

{% fragment liter, snippets.cc %}


### Iterators

An iterator is any object that, pointing to some element in a container, has the ability to iterate through the elements of that range using a set of operators.

* Minimum operators needed: increment (`++`) and dereference (`*`).
* A pointer is the simplest iterator.
* Brings some container-independence.
    * especially when using  `typedef`

<http://www.cplusplus.com/reference/iterator/>


### Pairs

{% fragment pairs, snippets.cc %}

### Tuples

{% fragment tuples, snippets.cc %}


### Assignment

* Download a human genome file from <ftp://ftp.ensembl.org/pub/release-87/fasta/homo_sapiens/dna/> .
This is a sequence of characters from the dictionary `{A,C,G,T,N}`.
* List all the possible 3-letter combinations (k-mers for k=3) that appear in this file together with the number of appearances of each in the file, in order of number of appearances.
    * Output should be something like

```
NNN 3345028
AGT 2348
CTT 1578
...
```

* Hint: use only associative containers. You'll need to extend their functionality to some kind of counter.

{% endidio %}
