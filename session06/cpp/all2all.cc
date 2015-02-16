#include <mpi.h>
#include <iostream>
#include <cassert>

/// "functions"
std::string root_sends(std::string const message, int const size) {
    assert(message.size() > size);
    int N = message.size() / size;
    char result[256];
    int const error = MPI_Scatter(
            (void*) message.c_str(), N, MPI_CHAR,
            result, 256, MPI_CHAR, 0, MPI_COMM_WORLD
    );
    if(error != MPI_SUCCESS) throw;
    return result;
}

std::string others_get_it() {
    char buffer[256];
    int const error = MPI_Scatter(
            NULL, -1, MPI_CHAR, // not significant outside root
            buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD
    );
    if(error != MPI_SUCCESS) throw;
    return buffer;
}

/// "dummy"
int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    /// "main"
    std::string message = rank == 0 ?
        root_sends("This message is going to come out in separate channels", size):
        others_get_it();

    for(int i(0); i < size; ++i) {
        if(rank == i)
            std::cout << rank << ": " << message << "\n";
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /// "dummy"
    MPI_Finalize();
    return 0;
}
