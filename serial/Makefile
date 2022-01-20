# CC = mpicc
CC = cc
CCFLAGS = -Wall -Wextra 
CCFLAGS_D = -DDEBUG -Wall -Wextra -g -fsanitize=address -fno-omit-frame-pointer

SRC = serial_kd_tree.c \
		partition_test.c

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