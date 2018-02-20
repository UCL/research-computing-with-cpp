#include <string>
#include <vector>
#include <mpi.h>

typedef double density;
typedef double distance;
typedef double filling;

class Smooth {
public:
  Smooth(int sizex = 100, int sizey = 100, distance inner = 21.0, filling birth_1 = 0.278,
         filling birth_2 = 0.365, filling death_1 = 0.267, filling death_2 = 0.445,
         filling smoothing_disk = 0.147, filling smoothing_ring = 0.028);
  int Size() const;
  int Sizex() const;
  int Sizey() const;
  int Range() const;
  int Frame() const;
  const std::vector<density> &Field() const;
  void Field(std::vector<density> const &input);

  //! \brief Piecewise linear function defining the disk
  //! \details
  //! - 1 inside the disk
  //! - 0 outside the disk
  //! - 0 < x < 1 in the smoothing region
  double Disk(distance radius) const;
  //! \brief Piecewise linear function defining the ring
  //! \details
  //! - 1 inside the ring
  //! - 0 outside the ring
  //! - 0 < x < 1 in the smoothing region
  double Ring(distance radius) const;
  /// "Sigmoid_Signature"
  //! Smooth step function: 0 at -infty, 1 at +infty
  static double Sigmoid(double variable, double center, double width);
  /// end
  //! $e^{-4x / width}$: 0 at -infty, 1 at +infty
  static double Sigmoid(double x, double width);
  density Transition(filling disk, filling ring) const;
  //! Find the distance between 2 indices on a 1d torus
  int TorusDistance(int x1, int x2, int size) const;
  //! Find the Euclidean distance between points on a torus
  double Radius(int x1, int y1, int x2, int y2) const;
  //! Value of the integral over a single ring
  double NormalisationRing() const;
  //! Value of the integral over a single disk
  double NormalisationDisk() const;
  //! Sets the playing field to random values
  void SeedRandom();
  //! Sets the playing field to constant values
  void SeedConstant(density constant = 0);
  //! Adds a disk to the playing field
  void AddDisk(int x0 = 0, int y0 = 0);
  //! Adds a ring to the playing field
  void AddRing(int x0 = 0, int y0 = 0);
  //! Sets a single pixel in the field
  void AddPixel(int x0, int y0, density value);
  //! Moves to next step
  void Update();
  //! Prints current field to standard output
  void Write(std::ostream &out);

  //! Returns {disk, ring} integrals at point (x, y)
  std::pair<density, density> Integrals(int x, int y) const;

  //! Linear index from cartesian index
  int Index(int i, int j) const;
  //! Cartesian index from linear index
  std::pair<int, int> Index(int i) const;

private:
  int sizex, sizey;
  std::vector<density> field, work_field;
  filling birth_1, death_1;
  filling birth_2, death_2;
  filling smoothing_disk, smoothing_ring;
  distance inner, outer, smoothing;
  int frame;
  double normalisation_disk, normalisation_ring;

#ifdef HAS_MPI
public:
  MPI_Comm const &Communicator() const { return communicator; }
  void Communicator(MPI_Comm const &comm) { communicator = comm; }

  //! Update which layers computation and communication
  void LayeredUpdate();

  //! Figure start owned sites for given rank
  static int OwnedStart(int nsites, int ncomms, int rank);

  //! \brief Syncs fields between processes
  //! \details Assumes that each rank owns the sites given by OwnedRange.
  static void WholeFieldBlockingSync(std::vector<density> &field, MPI_Comm const &comm);

  //! \brief Syncs fields between processes without blocking
  //! \details Assumes that each rank owns the sites given by OwnedRange.
  static MPI_Request WholeFieldNonBlockingSync(std::vector<density> &field, MPI_Comm const &comm);
private:
  MPI_Comm communicator;
#endif
};
