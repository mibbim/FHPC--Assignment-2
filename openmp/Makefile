CC = gcc
# CC = cc
CCFLAGS = -Wall -Wextra -fopoenmp
CCFLAGS_D = -DDEBUG -Wall -Wextra -g -fopenmp

SRC = openmp_kdtree.c

EXE = $(SRC:.c=.x)

# eliminate default suffixes
.SUFFIXES:
SUFFIXES =

# just consider our own suffixes
.SUFFIXES: .c .x

all: $(EXE)

.PHONY: all

%.x: %.c 
	$(CC) $< -o $@ $(CCFLAGS)


debug: $(EXE)

.PHONY: debug

%.x: %.c 
	$(CC) $< -o $@ $(CCFLAGS_D)



clean:
	rm -f $(EXE) *~

.PHONY: clean