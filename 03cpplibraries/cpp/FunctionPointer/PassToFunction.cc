#include <stdio.h>  
#include <math.h>  
double integrate(double (*funcp)(double), double lo, double hi) {
  double  sum = 0.0;
  for (int i = 0;  i <= 100;  i++)
  {
    sum += (*funcp)(i / 100.0 * (hi - lo) + lo);
  }
  return sum / 100.0;
}
int main(void) {
  double  (*fp)(double) = sin;
  printf("sum(sin): %f\n", integrate(fp, 0.0, 1.0));
  return 0;
}
