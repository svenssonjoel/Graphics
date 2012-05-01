C99 = gcc -std=c99 

LIBS = -lglut -lGLEW -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

INC = -I../../
GFX = ../../Graphics/graphlib.o

all: show_feig

show_feig: show_feig.c 
	$(C99) $(CFLAGS) -o show_feig show_feig.c $(LIBS) $(INC) $(GFX)


clean: 
	rm show_feig