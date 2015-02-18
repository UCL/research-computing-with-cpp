#include "saxpy.h"

int saxpy(int n, float a, const float * x, int incx, float * y, int incy) {
    if (n < 0)
        return 1;

    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];

    return 0;
}
