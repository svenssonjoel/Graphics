
C99 = gcc -std=c99 
##-lglut -lGLEW -lGLU -lm -l3ds
LIBS = -lglut -lGLEW -lGL -lGLU -lm -l3ds -lpng
CFLAGS = -Wall

SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

%.o:%.c
	$(C99) -c -o $@ $^ $(CFLAGS) $(LIBS)


all: graphlib.o


graphlib.o: $(OBJS)
	ld -r -o graphlib.o $(OBJS)

clean: 
	rm *.o
