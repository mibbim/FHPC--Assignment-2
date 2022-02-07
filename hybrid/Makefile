
SRC = hybrid_kd_tree.c
CC = mpicc
CFLAGS = -Wall -Wextra -fopenmp
CCFLAGS_D = -DDEBUG -Wall -Wextra -g

EXE = $(SRC:.c=.x)


default:
	$(CC) $(SRC) -o $(EXE) $(CFLAGS)

debug:
	$(CC) $(SRC) -o $(EXE) $(CFLAGS_D)

clean:
	rm $(EXE)