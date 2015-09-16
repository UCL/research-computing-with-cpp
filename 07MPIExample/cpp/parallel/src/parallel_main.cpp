#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <ctime>
#include "Smooth.h"
#include "ParallelWriter.h"
#include "SingleWriter.h"
#include "TextWriter.h"
#include "BinaryWriter.h"
#include "XdrWriter.h"
#include <mpi.h>


void report_time(std::ostream & stream, std::string text, std::clock_t end, std::clock_t start) {
  stream << text << " " << (end-start)/static_cast<double>(CLOCKS_PER_SEC) << std::endl;
}

int main(int argc, char **argv){
  MPI_Init (&argc, &argv);
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  
  std::string config_path = argv[1];
  std::ifstream config_file(config_path.c_str());
  
  int width;
  int height;
  int range;
  int spots;
  int frames;

  std::string label;
  config_file >> label >> width;
  assert(label=="width:");
  config_file >> label >> height;
  assert(label=="height:");
  config_file >> label >> range ;
  assert(label=="range:");
  config_file >> label >> frames;
  assert(label=="frames:");
  config_file >> label >> spots ;
  assert(label=="spots:");
  
  std::ostringstream report_name;
  report_name << "report" << rank << ".yml" << std::flush;
  std::ofstream report(report_name.str().c_str());  
  
  report << "rank: " << rank << std::endl;
  report << "range: " << range << std::endl;
  report << "width: " << width << std::endl;
  report << "height: " << height << std::endl;
  report << "frames: " << frames << std::endl;
  report << "spots: " << spots << std::endl;

  std::clock_t start=std::clock();
  Smooth smooth(width,height,range,rank,size);
  std::clock_t setup=std::clock();
  
  for (int i=0;i<spots;i++){
    smooth.SeedRandomDisk();
  }
  std::clock_t seed=std::clock();
  ParallelWriter writer(smooth, rank, size);
  writer.Header(frames);
  
  std::cout << "Rank " << rank << "ready" << std::endl;
  
  std::vector<std::clock_t> frame_times(frames+1);
  
  for (unsigned int frame=0; frame<frames; frame++) {
    frame_times[frame]=std::clock();
    writer.Write();
    smooth.UpdateAndCommunicateAsynchronously();
    std::cout << "Rank " << rank << " completed frame: " << smooth.Frame() << std::endl;
  }
  frame_times[frames]=std::clock();
  
  report_time(report, "setup:", setup, start);
  report_time(report, "seed:", seed, setup);
  report_time(report,"all_frames: ", frame_times[frames], frame_times[0]);
  
  report << "frames:" << std::endl;
  
  for (unsigned int frame=0; frame<frames; frame++) {
    report_time(report, "    -", frame_times[frame+1], frame_times[frame]);
  }
  writer.Close();
  MPI_Finalize();
}
