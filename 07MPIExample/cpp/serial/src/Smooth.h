#include <string>
#include <vector>

typedef double density;
typedef double distance;
typedef double filling;

class Smooth {
public:
  Smooth(int sizex = 100, int sizey = 100, distance inner = 21.0, filling birth_1 = 0.278,
         filling birth_2 = 0.365, filling death_1 = 0.267, filling death_2 = 0.445,
         filling smoothing_disk = 0.147, filling smoothing_ring = 0.028);
  int Size();
  int Sizex();
  int Sizey();
  int Range();
  const std::vector<std::vector<density>> &Field() const;
  double Disk(distance radius) const;
  double Ring(distance radius) const;
  static double Sigmoid(double variable, double center, double width);
  density transition(filling disk, filling ring) const;
  int TorusDifference(int x1, int x2, int size) const;
  double Radius(int x1, int y1, int x2, int y2) const;
  double NormalisationRing() const;
  double NormalisationDisk() const;
  filling FillingRing(int x, int y) const;
  filling FillingDisk(int x, int y) const;
  density NewState(int x, int y) const;
  void SeedRandom();
  void SeedDisk();
  void SeedRing();
  void Update();
  void QuickUpdate();
  void Write(std::ostream &out);
  int Frame() const;

private:
  int sizex;
  int sizey;
  std::vector<std::vector<density>> field1;
  std::vector<std::vector<density>> field2;
  std::vector<std::vector<density>> *field;
  std::vector<std::vector<density>> *fieldNew;
  distance inner;
  filling birth_1;
  filling birth_2;
  filling death_1;
  filling death_2;
  filling smoothing_disk;
  filling smoothing_ring;
  distance outer;
  distance smoothing;
  int frame;
  double normalisation_disk;
  double normalisation_ring;
};
