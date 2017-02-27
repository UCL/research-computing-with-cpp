#include <mpi.h>
#include "SmoothWriter.h"

class ParallelWriter : public SmoothWriter {
public:
  ParallelWriter(Smooth &smooth, int rank, int size);
  ~ParallelWriter();
  void Write();
  void Header(int frames);
  void Close();

private:
  MPI_File outfile;
};
