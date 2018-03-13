#ifndef SAXPY_H
#define SAXPY_H

int saxpy(int, float, const float *restrict, int, float *restrict, int);
int saxpy_fast(int, float, const float *restrict, int, float *restrict, int);

#endif
