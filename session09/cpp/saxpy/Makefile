CC = gcc-4.9
CFLAGS = -std=c99 -O2

.PHONY: all clean

all: saxpy

clean:
	rm -f saxpy main.o saxpy.o saxpy_fast.o

saxpy: main.o saxpy.o saxpy_fast.o

saxpy_fast.o: CFLAGS += -ftree-loop-vectorize -fopt-info-vec
