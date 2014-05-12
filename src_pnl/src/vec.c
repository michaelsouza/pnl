#include "pnl.h"

Vec *vec_new(PnlInt n){
	Vec *x = PnlMalloc(1,sizeof(Vec));
	x->x = PnlMalloc(n,sizeof(PnlScalar));
	return x;
}

void *vec_free(Vec *x){
	PnlFree(x->x);
	PnlFree(x);
	return NULL;
}
