#include <ctime>
#include <fstream>
#include <iostream>
#include "Smooth.h"

void report(std::ostream &stream, std::string text, std::clock_t end, std::clock_t start) {
  stream << text << " " << (end - start) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
}

int main(int argc, char **argv) {
  std::ofstream timings("timings.yml");
  std::clock_t start = std::clock();
  Smooth smooth(50);
  std::clock_t setup = std::clock();
  smooth.SeedRing();
  std::clock_t seed = std::clock();
  unsigned int frames = 10;
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
