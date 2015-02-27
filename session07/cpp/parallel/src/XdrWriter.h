#include "SmoothWriter.h"
#include <cstdio>
#include <rpc/types.h>
#include <rpc/xdr.h>

class XDRWriter: public SmoothWriter{
  public:
    XDRWriter(Smooth & smooth, int rank, int size);
    ~XDRWriter();
    void Write();
    void Header(int frames);
  private:
    XDR xdrfile;
};
