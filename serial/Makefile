SRC = serial_kd_tree.c 

CC =  mpicc
CFLAGS = -O3
EXE:=$(SRC:.c=.x)


# eliminate default suffixes
.SUFFIXES:
SUFFIXES =

# just consider our own suffixes
.SUFFIXES: .c .x

all: $(EXE)
.PHONY: all
%.x: %.c
	$(CC) $< -o $@ $(CFLAGS)


clean:
.PHONY: clean
	rm -f $(EXE) *~

