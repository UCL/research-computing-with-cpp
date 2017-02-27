#include <cstdio>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include "SmoothWriter.h"

class SingleWriter : public SmoothWriter {
public:
  SingleWriter(Smooth &smooth, int rank, int size);
  ~SingleWriter();
  void Write();
  void Header(int frames);

private:
  XDR xdrfile;
};
