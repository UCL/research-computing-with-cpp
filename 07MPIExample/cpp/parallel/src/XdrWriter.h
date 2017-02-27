#include "SmoothWriter.h"
/// "Includes"
#include <cstdio>
#include <rpc/types.h>
#include <rpc/xdr.h>
/// "Class"
class XDRWriter : public SmoothWriter {
public:
  XDRWriter(Smooth &smooth, int rank, int size);
  ~XDRWriter();
  void Write();
  void Header(int frames);

private:
  XDR xdrfile;
};
