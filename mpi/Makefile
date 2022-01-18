CC = mpicc
# CC = cc
CCFLAGS = -Wall -Wextra 
CCFLAGS_D = -DDEBUG -Wall -Wextra -g

SRC = mpi_kd_tree.c \
	master_mpi_test.c \
	type_mpi_test.c

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