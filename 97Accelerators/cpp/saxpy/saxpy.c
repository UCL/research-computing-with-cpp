#include "saxpy.h"

/// "saxpy"
int saxpy(int n, float a, const float * restrict x, int incx,
		  float * restrict y, int incy) {
    if (n < 0)
        return 1;

    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];

    return 0;
}
