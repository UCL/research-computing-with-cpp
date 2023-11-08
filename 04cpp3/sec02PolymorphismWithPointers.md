# Revisiting Polymorphism using Pointers

Early in the course we saw how we can use inheritance to create sub-types which are substitutable in place of a base type. Recall that:
- In order to have make use of different function implementations for objects which were created as base or derived classes, we need to use `virtual` functions. 
- `virtual` functions are looked up from a virtual table stored as part of the object's data. 
- If we cast to the base class then we will construct a fresh object with a new virtual table for the base class. We therefore needed to pass these objects by reference in order to use the derived class's virtual functions. 

One problem with using references for runtime polymorphism is that references cannot be placed in containers. We cannot define, for example, 
```cpp
vector<Shape &> shapes;   // Invalid code!
```
However we can define containers of **pointers**, such as:
```cpp
vector<unique_ptr<Shape>> shapes;
vector<shared_ptr<Shape>> shapes;
vector<weak_prt<Shape>> shapes;
vector<Shape *> shapes;
```
Each of these are valid declarations. Containers holding polymorphic types is a very common use of runtime polymorphism, as it allows us to write code which processes lists of objects without separating them out into separate lists for each sub-type. 

For example, if we want to render shapes to the screen, it makes sense to have a vector of shapes in our scene and draw them one by one:
```cpp
void drawScene(const vector<Shape *> shapes)
{
    for(const Shape * const shape : shapes)
    {
        shape->draw();
    }
}
```
- This code is now agnostic about what different kind of shapes exist. If we had to have a different container for each kind of shape, this code would be tightly coupled to any code which defines new shapes which is added or removed. This also avoids repetitive code doing the same thing on different containers. 
- It's not unusual for a method like `draw` to draw each new shape on top of any previous shapes that it overlaps; this would mean that we would want the `shapes` vector to be sorted in the correct order based on what shapes we want to be on top and which underneath. This would be more difficult to achieve if we had to have different containers for each type as we'd have to impose the ordering some other way. 
- The use of `const Shape * const` means that `shape` is a constant pointer to constant data i.e. this loop cannot change the pointer in the vector _or_ the object data that it refers to. Using `const` appropriately when working with pointers will make code safer! 

