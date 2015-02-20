// Next line tells CATCH we will use our own main function
#define CATCH_CONFIG_RUNNER
#include <cmath>
#include "catch.hpp"
#include "Smooth.h"
#include <mpi.h>


int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}


TEST_CASE ("MPI Tests"){
    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    REQUIRE(size==2); // test designed for a two-process situation
    SECTION("Basic ring communication works"){
       Smooth smooth(200,100,5,rank,size);
       smooth.SeedDisk(); // Half the Seeded Disk falls in smooth2's domain, so total filling will be half a disk.
       if (rank==0) {
         REQUIRE(abs(smooth.FillingDisk(15,0)-0.5)<0.1);
       }
       if (rank==1) {
         REQUIRE(smooth.FillingDisk(84,0)==0.0);
       }
       smooth.CommunicateMPI();
       if (rank==0) {
         REQUIRE(abs(smooth.FillingDisk(0,0)-0.5)<0.1);
       }
       if (rank==1) {
         REQUIRE(abs(smooth.FillingDisk(84,0)-0.5)<0.1);
       }
    }

    SECTION("Behaviour unchanged by parallelisation"){
       Smooth smooth(100,100,5,0,1);
       smooth.SeedRing();
       smooth.SeedRing(50,50);
    
       Smooth paraSmooth(100,100,5,rank,size);
       paraSmooth.SeedRing();
       paraSmooth.SeedRing(50,50);
       
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }
       
       smooth.QuickUpdate();
       paraSmooth.CommunicateMPI();
       paraSmooth.QuickUpdate();
       paraSmooth.CommunicateMPI();
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }
    }
    
    SECTION("Behaviour unchanged by parallelisation with derived datatype"){
       Smooth smooth(100,100,5,0,1);
       smooth.SeedRing();
       smooth.SeedRing(50,50);
    
       Smooth paraSmooth(100,100,5,rank,size);
       paraSmooth.SeedRing();
       paraSmooth.SeedRing(50,50);
       paraSmooth.CommunicateMPIDerivedDatatype();
       
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }
       smooth.QuickUpdate();
       paraSmooth.QuickUpdate();
       paraSmooth.CommunicateMPIDerivedDatatype();
      
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }

    }
  
    SECTION("Behaviour unchanged by asynchronous communication"){
       Smooth smooth(100,100,5,0,1);
       smooth.SeedRing();
       smooth.SeedRing(50,50);
    
       Smooth paraSmooth(100,100,5,rank,size);
       paraSmooth.SeedRing();
       paraSmooth.SeedRing(50,50);
       paraSmooth.CommunicateAsynchronously();
       
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }
       smooth.QuickUpdate();
       paraSmooth.UpdateAndCommunicateAsynchronously();
      
       for (unsigned int x=0;x<50;x++){
         for (unsigned int y=0;y<100;y++){
           REQUIRE(std::abs(smooth.Field(x+15+rank*50,y) - paraSmooth.Field(x+15,y))<0.00001);
         }
       }

    }

}
