#include <sstream>
#include <fstream>
#include "SmoothWriter.h"

SmoothWriter::SmoothWriter(Smooth & smooth, int rank, int size)
    :smooth(smooth),rank(rank),size(size){
     total_element_count=smooth.Sizex()*smooth.Sizey();
     local_element_count=smooth.LocalXSize()*smooth.Sizey();
     sizey=smooth.Sizey();
     sizex=smooth.Sizex();
     fname << "frames.dat" << std::flush;
};

