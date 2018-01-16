#include <stdio.h>  /* for printf */
double cm_to_inches(double cm) {
  return cm / 2.54;
}
int main(void) {
  double (*func1)(double)            = cm_to_inches;
  printf("Converting %f cm to %f inches by calling function.\n", 5.0, cm_to_inches(5.0));
  printf("Converting %f cm to %f inches by deref pointer.\n", 15.0, func1(15.0));
  return 0;
}
