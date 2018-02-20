#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "smooth.h"

Smooth::Smooth(int sizex, int sizey, distance inner, filling birth_1, filling birth_2,
               filling death_1, filling death_2, filling smoothing_disk, filling smoothing_ring)
    : sizex(sizex), sizey(sizey), field(sizex * sizey), work_field(sizex * sizey), inner(inner),
      birth_1(birth_1), birth_2(birth_2), death_1(death_1), death_2(death_2),
      smoothing_disk(smoothing_disk), smoothing_ring(smoothing_ring), outer(inner * 3),
      smoothing(1.0)
#ifdef HAS_MPI
      ,
      communicator(MPI_COMM_SELF)
#endif
{
  normalisation_disk = NormalisationDisk();
  normalisation_ring = NormalisationRing();
}

const std::vector<density> &Smooth::Field() const { return field; };
void Smooth::Field(std::vector<density> const &input) {
  assert(field.size() == input.size());
  field = input;
}

int Smooth::Range() const { return outer + smoothing / 2; }

int Smooth::Sizex() const { return sizex; }
int Smooth::Sizey() const { return sizey; }
int Smooth::Size() const { return sizex * sizey; }

/// "Disk_Smoothing"
double Smooth::Disk(distance radius) const {
  if(radius > inner + smoothing / 2)
    return 0.0;
  if(radius < inner - smoothing / 2)
    return 1.0;
  return (inner + smoothing / 2 - radius) / smoothing;
}
/// end

double Smooth::Ring(distance radius) const {
  if(radius < inner - smoothing / 2)
    return 0.0;
  if(radius < inner + smoothing / 2)
    return (radius + smoothing / 2 - inner) / smoothing;
  if(radius < outer - smoothing / 2)
    return 1.0;
  if(radius < outer + smoothing / 2)
    return (outer + smoothing / 2 - radius) / smoothing;
  return 0.0;
}

double Smooth::Sigmoid(double variable, double center, double width) {
  return Sigmoid(variable - center, width);
}
double Smooth::Sigmoid(double x, double width) { return 1.0 / (1.0 + std::exp(-4.0 * x / width)); }

density Smooth::Transition(filling disk, filling ring) const {
  auto const sdisk = Sigmoid(disk - 0.5, smoothing_disk);
  auto const t1 = birth_1 * (1.0 - sdisk) + death_1 * sdisk;
  auto const t2 = birth_2 * (1.0 - sdisk) + death_2 * sdisk;
  return Sigmoid(ring - t1, smoothing_ring) * Sigmoid(t2 - ring, smoothing_ring);
}

int Smooth::Index(int i, int j) const { return i * Sizex() + j; }
std::pair<int, int> Smooth::Index(int i) const { return {i / Sizex(), i % Sizex()}; }

/// "Torus_Difference"
int Smooth::TorusDistance(int x1, int x2, int size) const {
  auto const remainder = std::abs(x1 - x2) % size;
  return std::min(remainder, std::abs(remainder - size));
}
/// end

double Smooth::Radius(int x1, int y1, int x2, int y2) const {
  int xdiff = TorusDistance(x1, x2, sizex);
  int ydiff = TorusDistance(y1, y2, sizey);
  return std::sqrt(xdiff * xdiff + ydiff * ydiff);
}

double Smooth::NormalisationDisk() const {
  double total = 0.0;
  for(int x = 0; x < sizex; x++)
    for(int y = 0; y < sizey; y++)
      total += Disk(Radius(0, 0, x, y));
  return total;
}

double Smooth::NormalisationRing() const {
  double total = 0.0;
  for(int x = 0; x < sizex; x++)
    for(int y = 0; y < sizey; y++)
      total += Ring(Radius(0, 0, x, y));
  return total;
}

void Smooth::Update() {
#ifdef HAS_MPI
  int rank, ncomms;
  MPI_Comm_rank(Communicator(), &rank);
  MPI_Comm_size(Communicator(), &ncomms);

  WholeFieldBlockingSync(field, communicator);
  auto const start = OwnedStart(Size(), ncomms, rank);
  auto const end = OwnedStart(Size(), ncomms, rank + 1);
#else
  auto const start = 0;
  auto const end = field.size();
#endif

  for(int i(start); i < end; ++i) {
    auto const xy = Index(i);
    auto const integrals = Integrals(xy.first, xy.second);
    work_field[i] = Transition(integrals.first, integrals.second);
  }

  std::swap(field, work_field);
  frame++;
}

#ifdef HAS_MPI
void Smooth::LayeredUpdate() {
  int rank, ncomms;
  MPI_Comm_rank(Communicator(), &rank);
  MPI_Comm_size(Communicator(), &ncomms);

  // Start synchronising the field
  auto request = WholeFieldNonBlockingSync(field, communicator);
  auto const start = OwnedStart(Size(), ncomms, rank);
  auto const end = OwnedStart(Size(), ncomms, rank + 1);
  // Determine which locations could be affected by other processes
  auto const interaction = Sizex() * static_cast<int>(std::floor(outer + smoothing / 2 + 1));

  // Define a lambda function to update part of our field
  auto const set_work_field_at_index = [this](int i) {
    auto const xy = Index(i);
    auto const integrals = Integrals(xy.first, xy.second);
    work_field[i] = Transition(integrals.first, integrals.second);
  };

  // Update those locations that cannot be affected by other processes
  for(int i(start + interaction); i < end - interaction; ++i)
    set_work_field_at_index(i);

  // Wait for field sync to finish
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Update the remaining locations
  for(int i(start); i < std::min(end, start + interaction); ++i)
    set_work_field_at_index(i);
  for(int i(std::min(end, end - interaction)); i < end; ++i)
    set_work_field_at_index(i);

  std::swap(field, work_field);
  frame++;
}
#endif

std::pair<density, density> Smooth::Integrals(int x, int y) const {
  density ring_total(0), disk_total(0);
  for(std::vector<density>::size_type i(0); i < field.size(); ++i) {
    auto const cartesian = Index(i);
    int deltax = TorusDistance(x, cartesian.first, sizex);
    if(deltax > outer + smoothing / 2)
      continue;

    int deltay = TorusDistance(y, cartesian.second, sizey);
    if(deltay > outer + smoothing / 2)
      continue;

    double radius = std::sqrt(deltax * deltax + deltay * deltay);
    double fieldv = field[i];
    ring_total += fieldv * Ring(radius);
    disk_total += fieldv * Disk(radius);
  }
  return {disk_total / NormalisationDisk(), ring_total / NormalisationRing()};
}

void Smooth::SeedRandom() {
  for(int x = 0; x < sizex; x++)
    for(int y = 0; y < sizey; y++)
      field[Index(x, y)] += (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
}

void Smooth::SeedConstant(density constant) { std::fill(field.begin(), field.end(), constant); }
void Smooth::AddDisk(int x0, int y0) {
  for(int x = 0; x < sizex; x++)
    for(int y = 0; y < sizey; y++)
      field[Index(x, y)] += Disk(Radius(x0, y0, x, y));
}

void Smooth::AddRing(int x0, int y0) {
  for(int x = 0; x < sizex; x++)
    for(int y = 0; y < sizey; y++)
      field[Index(x, y)] += Ring(Radius(x0, y0, x, y));
}

void Smooth::AddPixel(int x0, int y0, density value) { field[Index(x0, y0)] = value; }

void Smooth::Write(std::ostream &out) {
  for(int x = 0; x < sizex; x++) {
    for(int y = 0; y < sizey; y++)
      out << field[Index(x, y)] << " , ";
    out << std::endl;
  }
  out << std::endl;
}

int Smooth::Frame() const { return frame; }

#ifdef HAS_MPI
int Smooth::OwnedStart(int nsites, int ncomms, int rank) {
  assert(nsites >= 0);
  assert(ncomms > 0);
  assert(rank >= 0 and rank <= ncomms);
  return rank * (nsites / ncomms) + std::min(nsites % ncomms, rank);
}

void Smooth::WholeFieldBlockingSync(std::vector<density> &field, MPI_Comm const &comm) {
  int rank, ncomms;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ncomms);

  if(ncomms == 1)
    return;

  std::vector<int> displacements{0}, sizes;

  for(int i(0); i < ncomms; ++i) {
    displacements.push_back(Smooth::OwnedStart(field.size(), ncomms, i + 1));
    sizes.push_back(displacements.back() - displacements[i]);
  }

  MPI_Allgatherv(MPI_IN_PLACE, sizes[rank], MPI_DOUBLE, field.data(), sizes.data(),
                 displacements.data(), MPI_DOUBLE, comm);
}

MPI_Request Smooth::WholeFieldNonBlockingSync(std::vector<density> &field, MPI_Comm const &comm) {
  int rank, ncomms;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ncomms);

  std::vector<int> displacements{0}, sizes;

  for(int i(0); i < ncomms; ++i) {
    displacements.push_back(Smooth::OwnedStart(field.size(), ncomms, i + 1));
    sizes.push_back(displacements.back() - displacements[i]);
  }

  MPI_Request request;
  MPI_Iallgatherv(MPI_IN_PLACE, sizes[rank], MPI_DOUBLE, field.data(), sizes.data(),
                  displacements.data(), MPI_DOUBLE, comm, &request);
  return request;
}
#endif
