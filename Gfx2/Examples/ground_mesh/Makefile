C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGL -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: ground_mesh

ground_mesh: ground_mesh.c 
	$(C99) $(CFLAGS) -o ground_mesh ground_mesh.c $(GFX) $(LIBS) $(INC) 


clean: 
	rm ground_mesh
