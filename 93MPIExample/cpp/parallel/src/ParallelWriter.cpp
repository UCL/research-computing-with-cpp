#include "ParallelWriter.h"
#include <fstream>
#include <sstream>

void ParallelWriter::Write() {

  /// Seek
  int offset = 4 * sizeof(int) +                                      // The header
               rank * local_element_count * sizeof(double) +          // Offset within the frame
               smooth.Frame() * total_element_count * sizeof(double); // Frame offset in the file

  MPI_File_seek(outfile, offset, MPI_SEEK_SET);
  /// "Write"
  MPI_File_write(outfile, smooth.StartOfWritingBlock(), local_element_count, MPI_DOUBLE,
                 MPI_STATUS_IGNORE);
  /// "WriteEnd"
}

void ParallelWriter::Close() { MPI_File_close(&outfile); }

ParallelWriter::~ParallelWriter() {}

ParallelWriter::ParallelWriter(Smooth &smooth, int rank, int size)
    : SmoothWriter(smooth, rank, size) {
  /// "Open"
  MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(fname.str().c_str()),
                MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &outfile);
  /// "OpenEnd"
}

void ParallelWriter::Header(int frames) {
  if(rank != 0) {
    return;
  }
  MPI_File_write(outfile, &sizex, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_write(outfile, &sizey, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_write(outfile, &size, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_write(outfile, &frames, 1, MPI_INT, MPI_STATUS_IGNORE);
}
