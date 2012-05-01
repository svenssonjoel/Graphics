C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: space2

space2: space2.c 
	$(C99) $(CFLAGS) -o space2 space2.c $(LIBS) $(INC) $(GFX)


clean: 
	rm space2