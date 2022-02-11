CC = mpicc
# CC = cc
CCFLAGS = -Wall -Wextra -O3 -fopenmp 
CCFLAGS_D = -DDEBUG -Wall -Wextra -g -O3 -fopenmp

SRC = hybrid_kd_tree.c \
	 nov_hybrid_kd_tree.c
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