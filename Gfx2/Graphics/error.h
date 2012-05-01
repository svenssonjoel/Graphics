
#ifndef __ERROR_H
#define __ERROR_H

// Empty the error queue 
extern void error_flushAll();

extern void error_print(char *prefix);

// old. Stop using.
extern void ignoreErrors();

// old. Stop using
extern void printGLError(char *prefix);



#endif
