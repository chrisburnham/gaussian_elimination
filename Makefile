#
# Makefile for the CIS-4230 Linear Equations project.
#

NVCC=nvcc
NVCFLAGS=-c -m64 -arch=sm_60 -D_XOPEN_SOURCE=600 -O2 -I../spica/C
LD=nvcc
LDFLAGS=-m64 -arch=sm_60
SOURCES=solve_system.c \
        linear_equations.c \
        cuda_elim.cu
OBJECTS=solve_system.o \
		linear_equations.o \
		cuda_elim.o
EXECUTABLE=LinEq

%.o:	%.c
	$(NVCC) $(NVCFLAGS) $< -o $@

%.o:	%.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

$(EXECUTABLE):	$(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -lm -L../spica/C -lSpicaC -o $@

# File Dependencies
###################

solve_system.o:	solve_system.c linear_equations.h

linear_equations.o:	linear_equations.c linear_equations.h matrix_defs.h cuda_elim.h

cuda_elim.o: cuda_elim.cu matrix_defs.h cuda_elim.h

# Additional Rules
##################
clean:
	rm -f *.o *.bc *.s *.ll *~ $(EXECUTABLE) CreateSystem
