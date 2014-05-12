#ifndef __pnl__
#define __pnl__

#include <stdlib.h>

#define PnlInt int
#define PnlScalar double
#define PnlMalloc(a,b) (malloc(a*sizeof(b)))
#define PnlFree(a) (free(a))

typedef struct{
	PnlInt n;
	PnlScalar *x;
} Vec;


Vec *vec_new(PnlInt n);
void *vec_free(Vec *x);

#endif
