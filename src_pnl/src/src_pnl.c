#include "pnl.h"
#include "stdio.h"

int main(void) {
	Vec *x = vec_new(5);
	PnlScalar y = x->x[0];


	//printf("!!!Hello World!!! %f",x[0]); /* prints !!!Hello World!!! */

	x = vec_free(x);

	return EXIT_SUCCESS;
}
