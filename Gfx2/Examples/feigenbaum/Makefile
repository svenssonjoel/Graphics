C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGL -lGLU  -l3ds -lpng -lm
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: feigenbaum

feigenbaum: feigenbaum.c 
	$(C99) $(CFLAGS) -o feigenbaum feigenbaum.c $(GFX) $(LIBS) $(INC) 


clean: 
	rm feigenbaum
