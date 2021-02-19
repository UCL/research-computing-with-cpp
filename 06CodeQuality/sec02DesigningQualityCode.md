---
title: Designing Quality Code
---

Consider the age-old software-engineering principle:

> Don't repeat yourself (DRY)

So far on this course, we've applied this by placing common functionality into functions and, in some cases, classes. Only using functions is commonly called "procedural" programming and has been used for decades to keep things simple. As Kelly Johnson says,

> Keep it simple, stupid (KISS) -- Kelly Johnson

DRY and KISS are two fundamental software engineering principles which can help guide you in designing and writing quality software. However, KISS does not *necessarily* mean writing everything in a procedural style; if that were the case, other programming models wouldn't be so popular. When software becomes large or complex, just using functions isn't enough to KISS. This is why object-oriented programming (or OOP) is so popular; OOP principles are designed to reduce complexity. The four principles of OOP are:

- Encapsulation
  - Each object keeps its state private; other objects can only change it through public methods.
- Abstraction
  - Each object hides its implementation and exposes a public-facing interface
- Inheritance
  - Objects sharing functionality can inherit behaviour from a parent
- Polymorphism
  - Classes inherited from the same parent can all be treated *mostly* similarly

Let's explore these principles through an example; consider them applied to a real object, say, a car. A car **encapsulates** most of its "state" in a way it is hidden from the driver. The "state" includes things like the amount of petrol left, the number of miles driven or the current radio station. A driver cannot directly interact with this state but instead accesses it through its public interface, through the fuel-gauge, the milometer and changing the radio via its controls. In this way, the car has **abstracted** away the implementation of any of these controls and outputs. This is why a driver can sit down in most cars and immediately know how much fuel remains or how to change the radio, because the public interface doesn't change very much even though the underlying implementation of any of these controls might. For example, the volume knob on a radio at some point was used to control some analogue circuit. Now, the same knob is used as input to a digital circuit which changes the volume in its own way. Even though the implementation has significantly changed, the interface remains the same. In fact, this is an example of **polymorphism**, where many (very different) cars can be used in (nearly) the same way because they all use similar interfaces. This is because cars have **inherited** their interfaces from a common ancestor.

In code, this example might look like:
```
class Car {
  public:
    virtual float getFuelLeft() = 0; // pure virtual because there's no good default
    int getMilesDriven(); // Notice there's no set
    void increaseVolume() {
      radio.increaseVolume(); // Pass on the call to the radio
    }
    void decreaseVolume();
    void changeStation(int stationNumber);

  private:
    int milesDriven;
    Radio radio;
};

class DieselCar: public Car {
  public:
    float getFuelLeft() {
      return dieselRemaining;
    }

  private:
    float dieselRemaining;
};

class ElectricCar: public Car {
  public:
    float getFuelLeft() {
      return batteryLevelRemaining*efficiency;
    }

  private:
    float batteryRemaining;
    float efficiency;
};

void main() {
  // This is polymorphism - both cars are still of type Car
  Car *myDieselCar = new DieselCar;
  Car *myElectricCar = new ElectricCar;

  // We don't need to know *actual* type of car!
  std::cout << myDieselCar.getFuelLeft() << std::endl;
  std::cout << myElectricCar.getFuelLeft() << std::endl;
}
```

Notice a few things here:
- We've used a pure virtual function because we *need* a subclass to override this; there's just no good default option.
- We've used **composition** instead of inheritance to handle the radio; this means we can change the type of radio without having a new subclass of car!
- There's no `setMilesDriven` because the driver shouldn't be able to set that (this is **encapsulation**)
- The electric car has used **abstraction** to hide how the fuel level is really calculated from the driver.
