
SRC = hybrid_kd_tree.c
CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp -O3
CCFLAGS_D = -DDEBUG -Wall -Wextra -g -O3

EXE = $(SRC:.c=.x)


default:
	$(CC) $(SRC) -o $(EXE) $(CFLAGS)

debug:
	$(CC) $(SRC) -o $(EXE) $(CFLAGS_D)

clean:
	rm $(EXE)