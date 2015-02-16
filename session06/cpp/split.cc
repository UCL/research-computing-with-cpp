#include <mpi.h>
#include <iostream>

int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    /// "main"
    MPI_Comm apple_orange;
    bool is_apple = rank % 3 == 1;
    MPI_Comm_split(MPI_COMM_WORLD, is_apple ? 0: 1, rank, &apple_orange);

    int nrank, nsize;
    MPI_Comm_rank(apple_orange, &nrank);
    MPI_Comm_size(apple_orange, &nsize);

    if(is_apple)
       std::cout << "I am an apple, says process "
           << nrank << "/" << nsize << " (" << rank << "/" << size << ")\n";

    MPI_Barrier(MPI_COMM_WORLD);

    if(not is_apple)
       std::cout << "I am an orange, says process "
           << nrank << "/" << nsize << " (" << rank << "/" << size << ")\n";

    /// "dummy"
    MPI_Finalize();
    return 0;
}
