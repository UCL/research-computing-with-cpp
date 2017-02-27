#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include "Smooth.h"

void report(std::ostream &stream, std::string text, std::clock_t end, std::clock_t start) {
  stream << text << " " << (end - start) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::string config_path = argv[1];
  std::ifstream config_file(config_path.c_str());
  int width;
  int height;
  int range;
  int spots;
  int frames;
  std::string label;
  config_file >> label >> width;
  assert(label == "width:");
  config_file >> label >> height;
  assert(label == "height:");
  config_file >> label >> range;
  assert(label == "range:");
  Smooth smooth(width, height, range, rank, size);
  std::ofstream timings("timings.yml");
  std::clock_t start = std::clock();
  std::clock_t setup = std::clock();
  smooth.SeedRing();
  std::clock_t seed = std::clock();
  std::vector<std::clock_t> frame_times(frames + 1);
  for(unsigned int frame = 0; frame < frames; frame++) {
    frame_times[frame] = std::clock();
    smooth.QuickUpdate();
  }
  frame_times[frames] = std::clock();
  timings << "size: " << smooth.Size() << std::endl;
  timings << "range: " << smooth.Range() << std::endl;
  report(timings, "setup:", setup, start);
  report(timings, "seed:", seed, setup);
  report(timings, "all_frames: ", frame_times[frames], frame_times[0]);
  timings << "frames:" << std::endl;
  for(unsigned int frame = 0; frame < frames; frame++) {
    report(timings, "    -", frame_times[frame + 1], frame_times[frame]);
  }
}
