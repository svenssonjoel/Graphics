C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGL -lGLU -l3ds -lpng -lm
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: view3ds

view3ds: view3ds.c 
	$(C99) $(CFLAGS) -o view3ds view3ds.c $(INC) $(GFX)  $(LIBS)


clean: 
	rm view3ds	
