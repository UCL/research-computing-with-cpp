#include <fstream>
#include "BinaryWriter.h"

/// "Write"
void BinaryWriter::Write() {
  outfile->write(reinterpret_cast<char*>(smooth.StartOfWritingBlock()),
      local_element_count*sizeof(double));
}

/// "Close"
BinaryWriter::~BinaryWriter(){
  delete outfile; // closes it
}

BinaryWriter::BinaryWriter(Smooth & smooth, int rank, int size)
    :SmoothWriter(smooth,rank,size)
{
     fname << "." << rank << std::flush;    
     outfile=new std::ofstream(fname.str().c_str(),std::ios::binary);
}

/// "Header"
void BinaryWriter::Header(int frames){
  outfile->write(reinterpret_cast<char*>(&sizex),sizeof(int));
  outfile->write(reinterpret_cast<char*>(&sizey),sizeof(int));
  outfile->write(reinterpret_cast<char*>(&rank),sizeof(int));
  outfile->write(reinterpret_cast<char*>(&size),sizeof(int));
  outfile->write(reinterpret_cast<char*>(&frames),sizeof(int));
}
