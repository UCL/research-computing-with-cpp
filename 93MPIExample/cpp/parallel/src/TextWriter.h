#include "SmoothWriter.h"

class TextWriter : public SmoothWriter {
public:
  TextWriter(Smooth &smooth, int rank, int size);
  ~TextWriter();
  void Write();
  void Header(int frames);

private:
  std::ostream *outfile;
};
