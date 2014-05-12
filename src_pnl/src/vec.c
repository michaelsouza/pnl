#include "pnl.h"

Vec *vec_new(PnlInt n){
	Vec *x = pnl_malloc(1,sizeof(Vec));
	x->x = pnl_malloc(n,sizeof(PnlScalar));
	return x;
}

void *vec_free(Vec *x){
	pnl_free(x->x);
	pnl_free(x);
	return NULL;
}
