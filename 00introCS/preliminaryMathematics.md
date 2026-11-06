---
title: Preliminary Mathematics
---

# Background: Preliminary Mathematics

A certain amount of mathematics is necessary to be able to write and reason about programs, or to describe computational problems precisely. As a scientific computing course, we will be looking at examples from a range of scientific disciplines which will use some amount of mathematics as a common language. So, while this course is intended for a wide audience and no problem will require specific domain knowledge from a particular science, we do require that you understand the meaning of some common mathematical notations and the basics of what they are describing. **This is especially important for the assessments as we will draw upon this language and notation when describing problems.**

## Algebra and arithmetic

We expect you to be able to understand equations in elementary algebra including polynomials, exponentials, and logarithms, and to solve straight-forward equations involving these terms. See e.g. the notes on complexity in week 7 for an idea of the kind of use of elementary algebra in the course. 

You should understand:

- The meaning of _associative_ and _commutative_ operations.
- Sum notation $\sum_{i = 0}^{N} x_i = x_0 + x_1 + ... + x_N$.
- Production notation $\prod_{i=0}^{N} x_i = x_0 \times x_1 \times ... \times x_N$. 

## Sets and Functions

You should understand:

- What a set is, and notation for defining sets and set membership (e.g. $x \in A$). 
- What a function is, and the function notation $f: A \rightarrow B$ (a function called $f$ that maps elements of set $A$ to elements of set $B$).
- What function composition is, and the following notation that if $f: A \rightarrow B$ and $g: B \rightarrow C$, then $g \circ f: A \rightarrow C$ is a function where $(g \circ f)(x) = g(f(x))$. 
- What a one-to-one function (a.k.a a bijection) is, and the notation $f^{-1}$ for the inverse.
- Notation for numerical intervals:
  - $[a, b]$ denotes the "open interval", the range $a \le x \le b$ (includes the end points);
  - $(a, b)$ denotes the "closed interval", the range $a \lt x \lt b$ (excludes end points);
  - $[a, b)$ denotes the "half-open interval", the range $a \le x \lt b$ (excludes one end point). 
- Notation for, and meaning of, the following common sets:
  - $\mathbb{N}$, the natural numbers;
  - $\mathbb{Z}$, the integers;
  - $\mathbb{R}$, the real numbers;
  - $\mathbb{B}$, the booleans $\mathbb{B} = \{\text{True}, \text{False}\}$.

You are not expected to have any further background in set theory or be able to write set theoretic proofs. 

## Logic

You should understand:

- Boolean (true/false) values and variables.
- Notation for propositional logic/boolean formulae ($A \land B$, $A \lor B$, $A \implies B$, $A \iff B$,  $\lnot A$).
- Quantifiers ($\exists$ meaning "there exists", $\forall$ meaning "for all") in first order logic. 
- Notations for prositional formulae $P(x)$.

For example you should be able to understand $\forall x \in A . P(x) \implies Q(x)$ to mean "for any element $x$ in the set $A$, if $P(x)$ is true then $Q(x)$ is true", and $\exists y \in B . R(y)$ to mean "there exists and element $y$ in the set $B$ such that the formula $R(y)$ is true".

## Calculus

You should understand:

- The basic idea of integration and its notation $\int f(x) dx$. 
  - You do not need to be able to _do_ integration yourself, but you should be familiar with what it means and be able to understand a simple numerical integration method like the [trapezium rule](https://en.wikipedia.org/wiki/Trapezoidal_rule). 
- The basic idea of differentiation and its notation $\frac{df}{dx}$. 
  - You do not need to be able to do the differentiation itself but you should understand its meaning well enough to understand the basic numerical approximation $\frac{df}{dx} \approx \frac{f(x + \Delta_x) - f(x - \Delta_x)}{2 \Delta_x}$. 

## Probability and Statistics

You should understand:

- The idea of a probability value, and that probability is fixed between $0$ and $1$. 
- The meaning of, and how to calculate, the following:
  - mean,
  - median,
  - mode,
  - variance,
  - standard deviation.

## Vectors and Matrices

You should understand:

- What a vector and a matrix are.
- Vector addition / subtraction.
- Vector dot product $v \cdot w$ and how to calculate it.
- Matrix addition / subtraction.
- Matrix-matrix product $M N$ and how to calculate it. 
- Matrix-vector product $M v$ and how to calculate it. 
- Compatibility of matrices/vectors for different operations based on their dimensions (e.g. vector dot product require both vectors to be the same length). 
- Vector notation $v_i$ or $v[i]$ for an element of a vector. 
- Matrix notation $M_{ij}$ for an element of a matrix. 

## Graphs 

Graphs may be unfamiliar to those of you without a significant computing background, so we will give a brief explanation here as they are crucial in many computer science problems. 

A graph is a set of vertices, $V$, and a set of edges, $E$. Each element of the edge set is a pair of vertices $(v_i, v_j)$ denoting that $v_i$ is connected to $v_j$ by an edge. Graphs are represented visually in the following way:


There are a number of kind of graphs that are important to know about. 

- Simple graphs: these are plain graphs where there is at most one edge connecting any two vertices. 
- Multigraphs: these are graphs where there may be more than one edge connecting two vertices. **In general the word "graph" will refer to this more general kind of graph.**
- Labeled graphs: these are graphs where vertices and edges may also have a label carrying additional information. For example in a _flow network_ edges are labelled by their capacity. 
- Directed graphs: these are graphs where the edges are directed and so the ordering of the pair $(v_i, v_j)$ is important: $v_j$ can be reached from $v_i$ but not vice versa because the edge is directed from $v_i$ to $v_j$. 
- Directed acyclic graphs: these are directed graphs where there is no path of non-zero length, following the directions of the edges, from a vertex back to itself. 
- Trees: these are directed graphs where every vertex has at most one predecessor (vertex linked by an incoming edge), but may have multiple successors (vertices linked by an outgoing edges). In trees the predecessor is called a "parent" and the successors are called "children". A vertex with no children is sometimes called a "leaf", and a vertex with multiple children is sometimes called a "branch". Vertices are usually labelled with some kind of data. **These are arguably the most widely used graph structure in computing and are essential for many algorithms and data structures.**
- Binary trees: these are trees where each vertex has at most two children (usually called the left and right child). A common definition is that a vertex either has _no_ children (a leaf) or it has exactly two (a branch). Binary search trees are a very important example of binary trees that we will talk about on this course. 
- Linked lists: these are directed graphs where vertices are labelled with data and each vertex has at most one successor (but may have multiple predecessors). These are a common data structure for many algorithms processing lists of data but are less commonly used in languages like C++ due to the performance benefits of random access arrays (see the data structures notes for more information). 

You will not need to know any graph algorithms or results of graph theory by heart, but you should understand what a graph is, what vertices (sometimes called nodes) and edges are, and what a path from one vertex to another is. You can find information about how we represent graphs in the notes on data structures. 