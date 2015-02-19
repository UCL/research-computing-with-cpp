#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "saxpy.h"

#define ITERATIONS 1000

int main(int argc, char * argv[]) {
    int n = 10000, incx = 1, incy = 1;

    if (argc < 1 || argc > 4) {
        fprintf(stderr, "Usage: %s [n=10000 [incx=1 [incy=1]]]\n", argv[0]);
        return -1;
    }

    if (argc > 1) {
		if (sscanf(argv[1], "%d", &n) != 1) {
			fprintf(stderr, "Failed to parse n from '%s'\n", argv[1]);
			return -2;
		}

		if (argc > 2) {
			if (sscanf(argv[2], "%d", &incx) != 1) {
				fprintf(stderr, "Failed to parse incx from '%s'\n", argv[2]);
				return -2;
			}

			if (argc > 3) {
				if (sscanf(argv[3], "%d", &incy) != 1) {
					fprintf(stderr, "Failed to parse incy from '%s'\n", argv[3]);
					return -3;
				}
			}
		}
    }

    printf("n = %d, incx = %d, incy = %d\n", n, incx, incy);

    const float a = 1.5f;
    float * x = malloc(n * incx * sizeof(float));
    float * y = malloc(n * incy * sizeof(float));

    if (x == NULL || y == NULL) {
        fputs("Failed to allocate x and y.", stderr);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        x[i * incx] = rand() / RAND_MAX;
        y[i * incy] = rand() / RAND_MAX;
    }

    clock_t total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        clock_t t = clock();
        int error = saxpy(n, a, x, incx, y, incy);
        total += clock() - t;
        if (error != 0) {
            fprintf(stderr, "SAXPY error with parameter %d\n", error);
            return error;
        }
    }
    printf("saxpy: %.6fms\n", (float)(total * 1000)/(CLOCKS_PER_SEC * ITERATIONS));

    total = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        clock_t t = clock();
        int error = saxpy_fast(n, a, x, incx, y, incy);
        total += clock() - t;
        if (error != 0) {
            fprintf(stderr, "SAXPY error with parameter %d\n", error);
            return error;
        }
    }
    printf("saxpy_fast: %.6fms\n", (float)(total * 1000)/(CLOCKS_PER_SEC * ITERATIONS));

    free(x);
    free(y);

    return 0;
}
