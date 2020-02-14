#
# Makefile for the CIS-4230 Linear Equations project.
#

CC=gcc
CFLAGS=-c -Wall -pthread -std=c99 -D_XOPEN_SOURCE=600 -O2 -I../spica/C
LD=gcc
LDFLAGS=-pthread
SOURCES=solve_system.c \
        linear_equations.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=LinEq

%.o:	%.c
	$(CC) $(CFLAGS) $< -o $@

$(EXECUTABLE):	$(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -L../spica/C -lSpicaC -o $@

# File Dependencies
###################

solve_system.o:	solve_system.c linear_equations.h

linear_equations.o:	linear_equations.c linear_equations.h


# Additional Rules
##################
clean:
	rm -f *.o *.bc *.s *.ll *~ $(EXECUTABLE) CreateSystem
