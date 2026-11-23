---
title: Program Analysis
---

# Program Analysis

We've discussed earlier in this course the importance of maintaining safety properties in our code. In this section we will spend a bit of time to analyse some of the most important program properties in more detail, so that we can create code that is, with respect to certain properties, correct by design. 

## Object States: Invariants

We've seen invariants a few times in this course already, but we haven't discussed in detail how we determine that a class does satisfy an invariant, and how this relates to the model of computation that we are using in C++[^formalsemantics].

[^formalsemantics]: It's important to understand that the C++ language is under-specified, and therefore has a number of sources of _undefined behaviour_. We will assume in this section the absence of undefined behaviour; if undefined behaviour is present in the program, we technically cannot reason about the program at all. Furthermore, a fully formal approach to this topic would require a mathemetically formal semantics of the language in question, which doesn't really exist for C++. Nevertheless, we by now understand the execution of C++ well enough to know how it is handling our variables, and therefore can make an _argument_ (but not a _proof_) that an invariant is preserved. 

### State Spaces

For our purposes the **state** of a program at any point can be taken to be the value of all variables and data stored in all allocated memory in the program. The **state space** is the set of all possible states. We will write the state space as $\Sigma$, and a state as $\sigma \in \Sigma$. As you can imagine, for a large and complex program the states, and the state space, can be extremely large! Furthermore, for an _imperative language_ (like C++, see week 5 on Programming Paradigms) any line of code can in principle modify the state, and reasoning about what's going on in programs is in general extremely difficult. 

One of the advantages of the object oriented programming paradigm is that is divides the state into smaller components. If we want to reason about objects of a given class, we only need to worry about the object state space, rather than the program state space. An object's state space is determined by the value of all the member variables of that class, along with any additional memory/resources that it allocates. OOP helps us simplify further by restricting access to the variables that make up this state space: if we make all of our data private then **the only way to modify the state of an object is through its public interface**.

### State Conditions

We can write a condition on state as a logical formula $P(\sigma)$. For example, suppose we define a class that stores an array $\vec v$ of size $N$. We could have the condition that this array is sorted. If we were to write this out more formally, we could write:

$P(\sigma) \equiv \forall i,j \in [0,N). i < j \implies \vec v_i \le \vec v_j$.

You will not be expected to write out conditions in logical form in this way, but you do need to understand what is written here. This states that for all values of $i$ and $j$ that are greater than or equal to 0 and less than $N$ (i.e. all valid indices of the vector $\vec v$) then if $i$ is less than $j$ the value in the vector at index $i$ is less than or equal to the value in the vector at index $j$ (sorted from least to greatest). 

### State Transitions

When a command in a programming language is executed the state can change. This change of state is called a _state transition_. We will write $\sigma \rightarrow \sigma^\prime$ if there is a command that will cause the program in state $\sigma$ to change to the new state $\sigma^\prime$, and we will write $\sigma \rightarrow^c \sigma^\prime$ if the command $c$ specifically causes this transition. 

When dealing with an individual object the set of possible commands that can induce changes to the state is highly restricted. 

- Public variables can be directly manipulated. A variable should therefore only be public if our invariant does not depend on it.
- - Private variables that have been returned by pointer or reference without `const` protections can be modified. As with public variables this should only ever be the case if we have no invariants that depend on these members. 
- Member functions can modify private variables. Only _public_ member functions can be called from outside the class. Any private member function can only be executed as the result of a public member calling it. 
- The assignment operator can overwrite private variables. This is also a member function, but it is easy to overlook because its implementation is usually defined by default.

Provided that your relevant data is well protected we therefore **only have to concern ourselves with the action of member functions** to show that a class maintains an invariant when state transitions are applied. 

### Arguments of Invariance

The set of _initial_ states that an object can have is determined by the class constructor. For an invariant $P$ to be maintained, the constructor must leave the object in an initial state that satisfies $P$; a good constructor will therefore carefully initialise all data. 

We can then argue that a class preserves an invariant across all accessible states by showing the following:

$P(\sigma) \wedge (\sigma \rightarrow \sigma^\prime) \implies P(\sigma^\prime)$,

which we can write in plain English as "if the object is in a state $\sigma$ that satisfies $P$ and there is a state transition to another state $\sigma^\prime$, then $\sigma^\prime$ also satisfies $P$". 

To show that this is the case for a given class, we need to look at all the public member functions and check that none of these can cause a state transition that violates the invariant. 

To make a complete argument for invariance therefore, we must follow these steps:

- Show that $P(\sigma)$ is true for all initial states by analysing the constructor.
- Show for each public member that if $P$ is satisfied before calling the public member, then $P$ is still satisfied after calling the public member. 
- Show that any variables that are public / returned by non-const pointer or reference do not contribute to the invariant $P$. 

To clarify this process, let's return to the examples from week 2. 

### Example: A Sphere

We have a class that represents a sphere, and stores the radius, mass, and density. 

```cpp
class Ball
{
    public:
    Ball(std::array<double, 3> p, double r, double m): position(p), radius(r), mass(m)
    {
        setDensity();
    }

    std::array<double, 3> position;
    double getRadius(){return radius;}
    double getMass(){return mass;}
    double getDensity(){return density;}

    double setRadius(double r)
    {
        radius = r;
        setDensity();
    }

    double setMass(double m)
    {
        mass = m;
        setDensity();
    }

    private:
    void setDensity()
    {
        density =  3 * mass / (4 * M_PI * pow(radius, 3));
    }

    double radius;
    double mass;
    double density;
};
```

The state is defined by four variables, $\sigma = (r, m, \rho, \vec p)$. Physically, we know that density, radius, and mass are connected by $\rho = \frac{M}{V}$ and $V = \frac{4 \pi}{3}r^3$, so we want to maintain the following invariant:

$\rho = \frac{3m}{4 \pi r^3}$

up to double floating point precision. We can make the following argument that our class preserves this invariant:

1. We show that the initial states satisfy the invariant. The contructor can take any position, mass, and radius, so $r$ and $m$ can be set arbitrarily after construction. However, we call `setDensity` in the constructor, so $\rho$ is correctly set satisfying the invariant when the constructor completes. 
2. Now we need to show that all public members preserve the invariant.
   - The getters do not modify the state and return private variables by value, so that they cannot be modified. 
   - `setRadius` allows $r$ to be changed arbitrarily, and does not change $m$. $\rho$ is brought up to date with the change in $r$ by calling `setDensity`. 
   - `setMass` likewise can modify the mass but calls `setDensity` after updating the value of $m$ to keep $\rho$ consistent. 
3. We also need to take into account any public members variables. The position is a public member variable and can be updated arbitrarily, but since the invariant does not depend on it in any way these state transitions cannot violate the invariant.


### Example: A Sorted List

Suppose we want to write a class stores a list and guarantees that this list will always been sorted from least element to greatest. We might write a class that looks like this:

```cpp
template<typename T>
class SortedList
{
public:
    SortedList() {};

    void insert(T x)
    {
        for(std::size_t i =0; i < list.size(); i++)
        {
            T& y = lsit[i];
            if(x <= y)
            {
                list.insert(i, x);
            }
        }

        // reached end of list
        list.push_back(x);
    }

    void remove(std::size_t i)
    {
        list.erase(i);
    }

    T find(T x) const
    {
        ...
    }

private:
    std::vector<T> list;
};
```

We'll now argue that this class preserves the invariant:

- The only constructor creates a `SortedList` that stores an empty vector in `list` (since this is the default initialisation of a vector). It is vacuously true that an empty vector is sorted since there are no elements to compare. 
- There is no public member data that can be modified at all.
- The public member functions are:
  - `find`: although we have not included the definition here for brevity, this is a `const` member function and therefore does not modify the state. 
  - `remove`: this function does modify the state, by removing the element at position `i` of the vector. If `list` was sorted before calling `remove`, then it is still sorted after calling `remove`: if an element $x$ appears before an element $y$ in the new list, $x$ must have also appeared before $y$ in the original list, and therefore we must have $x \le y$. 
  - `insert`: this function modifies the state, by inserting a new element $x$.
    - If `list` is empty then $x$ is simply inserted as the first element of the list. A list with a single element is trivially sorted.  
    - The loop finds the position $i$ of the first element $y$ in the list such that $x \le y$, and inserts $x$ at position $i$. The list of elements before this point is unchanged (and therefore still sorted), and the list of elements after this point (including $y$) is shifted along by one, but none the less still sorted. The list is therefore still sorted provided that (1) the all elements preceding $i$ are $\le x$ and (2) all elements after $i$ are $\ge x$.
    - Since $y$ is the first value where $x \le y$ and the original list is sorted, it follows that all elements before this point are strictly $< x$, so condition (1) is true.
    - Since the original list was sorted, all elements after position $i$ are $\ge y$, and since $y \ge x$ we have that all elements appearing in the list after $x$ are $\ge x$, satisfying (2). 


## Object States: Pre-conditions and Post-conditions

The idea of class invariants can be straight-forwardly generalised by allowing the conditions satisfied by the state before and after the tramsition to be different. The condition before the transition is called a _pre-condition_, and the condition afterwards is the _post-condition_. For pre-condition $P$ and post-condition $Q$ we aim to show:

$P(\sigma) \implies Q(\sigma^\prime)$.

This does not require an inductive argument, since it only applies to transitions which occur starting from a state satisfying $P$; we don't need to show that all states satisfy $P$. 

For example, suppose that we have a class called `MonotonicInt`. The state of this class is just a single integer $z \in \mathbb{Z}$. This class needs to satisfy the following property:

$z = n \implies z^\prime \ge n$.

This property means that if the our `MonotonicInt` holds a value $z = n$, then any state that it can transition to must hold a state at least as large as $n$ (i.e. its value can only increase or stay the same, never decrease). 

We can show that this property is maintained by showing that for all allowable transitions defined by the class interface, the member variable $i$ never decreases.

```cpp
class MonotonicInt
{
public:
    MonotonicInt(int x) : z(x) {}

    int get() const {return z;}

    void inc()
    {
        capped_assignment(z + 1);
    }

    void add(int y)
    {
        if(y < 0)
        {
            throw std::runtime_error("Monotonic int cannot change by a negative number.");
        }

        capped_assignment(z + y);
    }

    MonotonicInt& operator=(const MonotonicInt& other) = delete;
    MonotonicInt& operator=(MonotonicInt&& other) = delete;

private:
    void capped_assignment(int x)
    {
        z = x >= z ? x : std::numeric_limits<int>::max();
    }

    int z;
};
```

An object of this class can be initialised in any state by the constructor. 

To show that our monotonic property is satisfied, we need to show that $z$ always increases. 

- The variable $z$ is private and cannot be directly modified. The getter only returns by value (a copy).  
- The public members each satisfy the property:
    - `get()` is const and therefore does not cause any state transition. 
    - `inc()` is monotonic since $z$ is set by `capped_assignment`, which either sets $z$ to a new value $x$ is if it $\ge z$ or to the maximum value of an int, which must be $\ge z$. 
    - `add(int y)` is monotonic since if $y$ is negative then an exception is thrown and $z$ is not updated and retains its original value. Otherwise it is updated by `capped_assignment` and therefore monotonic. 
    - The default assignment operators have been deleted, since they could otherwise violate the monotonicity of $z$ by copying in a lower value. 

Note that in this class we have checked that expressions like `z + 1` and `z + y` are greater than `z`: this is to guard against _integer overflow_, where an `int` becomes to large to represent and wraps around to being a negative number. Issues like this can make writing safe classes trickier than it seems at first glance! Unsigned integers types can also overflow/underflow, and floating point types can lead to `nan` or `inf` values under certain circumstances. 