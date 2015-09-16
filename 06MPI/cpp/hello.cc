#include <mpi.h>
#include <iostream>

int main(int argc, char * argv[]) {
    /// Must be first call
    MPI_Init (&argc, &argv);
    /// Now MPI calls possible

    /// Size of communicator and process rank
    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    std::cout << "Processor " << rank << " of " << size << " says hello\n";

    /// Must be last MPI call
    MPI_Finalize();
    /// No more MPI calls from here
    return 0;
}
