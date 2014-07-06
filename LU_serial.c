/* Serial algorithm implementation for LU decomposition */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.h"


int main(int argc, char * argv[])
{
    int X=atoi(argv[1]);
    int Y=X;
    double ** A=malloc2D(X,Y);
    int i,j,k;
    double l;
    struct timeval ts,tf;
    double total_time;
	
	init2D(A,X,Y);
	
	gettimeofday(&ts,NULL);
	for (k=0;k<X-1;k++)
		for (i=k+1;i<X;i++) {
			l=A[i][k]/A[k][k];
			for (j=k;j<Y;j++)		
				A[i][j]-=l*A[k][j];
		}
	gettimeofday(&tf,NULL);
	total_time=(tf.tv_sec-ts.tv_sec)+(tf.tv_usec-ts.tv_usec)*0.000001;
	printf("LU-Serial\t%d\t%.3lf\n",X,total_time);
    	char * filename="output_serial";
    	print2DFile(A,X,Y,filename);
	return 0;
}
