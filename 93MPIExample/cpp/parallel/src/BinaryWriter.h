#include "SmoothWriter.h"

class BinaryWriter : public SmoothWriter {
public:
  BinaryWriter(Smooth &smooth, int rank, int size);
  ~BinaryWriter();
  void Write();
  void Header(int frames);

private:
  std::ostream *outfile;
};
