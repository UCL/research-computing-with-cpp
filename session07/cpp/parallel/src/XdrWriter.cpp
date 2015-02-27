#include "XdrWriter.h"
/// "Write"
void XDRWriter::Write() {

  char * start_to_write=reinterpret_cast<char*>(smooth.StartOfWritingBlock());
  xdr_vector(&xdrfile,start_to_write,local_element_count,sizeof(double),reinterpret_cast<xdrproc_t>(xdr_double));
}
/// "Destroy"
XDRWriter::~XDRWriter(){
}

XDRWriter::XDRWriter(Smooth & smooth, int rank, int size)
    :SmoothWriter(smooth,rank,size)
{
     fname << "." << rank << std::flush;    
     std::string mode("w");
     std::FILE * myFile = std::fopen(fname.str().c_str(),mode.c_str());
     /// "Create"
     xdrstdio_create(&xdrfile, myFile, XDR_ENCODE);
     /// "EndCreate"
}

void XDRWriter::Header(int frames){
  xdr_int(&xdrfile,&sizex);
  xdr_int(&xdrfile,&sizey);
  xdr_int(&xdrfile,&rank);
  xdr_int(&xdrfile,&size);
  xdr_int(&xdrfile,&frames);

}

