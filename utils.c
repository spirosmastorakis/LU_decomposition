/*
 This file includes functions that implement:
 
 1) Allocation of a 2-dimensional matrix
 2) Random initialization of the allocated matrix elements
 3) Deallocation of the memory used by the 2-dimensional matrix
 4) Printing of the resulting matrix to standard output or file
 
 */

#include <stdio.h>
#include <stdlib.h>

double ** malloc2D(int X, int Y) {
 	int i,j;
	double **a;
	
	a=malloc(X*sizeof(double*));
    if (a==NULL) {
        fprintf(stderr,"Malloc failed!\n");
        exit(-1);
    }
	a[0]=calloc(X*Y,sizeof(double));
    if (a[0]==NULL) {
        fprintf(stderr,"Malloc failed!\n");
        exit(-1);
    }

	for (i=1;i<X;i++)
		a[i]=a[i-1]+Y;	
	return a;
}

void free2D(double ** a, int X, int Y) {
    int i;
    if (X>1) {
        for (i=1;i<X;i++)
            a[i]=NULL;
    }
    free(a[0]);
    free(a);
}

void init2D(double ** a, int X, int Y) {
    int i,j;
    for (i=0;i<X;i++)
        for (j=0;j<Y;j++)
            a[i][j]=(rand()%100000)/10000.0;
}

void print2D(double ** a, int X, int Y) {
    int i,j;
    for (i=0;i<X;i++) {
        for (j=0;j<Y;j++)
            printf("%lf ",a[i][j]);
        printf("\n");
    }
}

void print2DFile(double **a, int X, int Y, char * filename) {
    int i,j;
    FILE * f=fopen(filename,"w");
    for (i=0;i<X;i++) {
        for (j=0;j<Y;j++) 
            fprintf(f,"%lf ",a[i][j]);
        fprintf(f,"\n");
    }
    fclose(f);
}   
