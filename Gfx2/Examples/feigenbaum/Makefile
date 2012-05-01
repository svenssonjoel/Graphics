C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: feigenbaum

feigenbaum: feigenbaum.c 
	$(C99) $(CFLAGS) -o feigenbaum feigenbaum.c $(LIBS) $(INC) $(GFX)


clean: 
	rm feigenbaum