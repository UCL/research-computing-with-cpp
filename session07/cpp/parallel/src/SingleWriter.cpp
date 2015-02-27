#include <mpi.h>
#include "SingleWriter.h"

void SingleWriter::Write() {

  double * receive_buffer;

  if (rank==0){
     receive_buffer=new double[total_element_count];
  }
  
  MPI_Gather(smooth.StartOfWritingBlock(),local_element_count,MPI_DOUBLE,
      receive_buffer,local_element_count,MPI_DOUBLE,0,MPI_COMM_WORLD); 

  if (rank==0){
     xdr_vector(&xdrfile,reinterpret_cast<char*>(receive_buffer),
         total_element_count,sizeof(double),reinterpret_cast<xdrproc_t>(xdr_double));
  }
}

SingleWriter::~SingleWriter(){
}

SingleWriter::SingleWriter(Smooth & smooth, int rank, int size)
    :SmoothWriter(smooth,rank,size)
{
     if (rank!=0) {
       return;
     }
     std::string mode("w");
     std::FILE * myFile = std::fopen(fname.str().c_str(),mode.c_str());
     xdrstdio_create(&xdrfile, myFile, XDR_ENCODE);
}

void SingleWriter::Header(int frames){
  if (rank!=0) {
    return;
  }
  xdr_int(&xdrfile,&sizex);
  xdr_int(&xdrfile,&sizey);
  xdr_int(&xdrfile,&size);
  xdr_int(&xdrfile,&frames);

}
