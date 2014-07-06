/* MPI implementation using collective communication among processes.
 The matrix rows are distributed in a round-robin fashion among processes. */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include "utils.h"
#include <string.h>


int main (int argc, char * argv[]) {
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int X,Y,x,y,X_ext,i,j,k,dif,srank;
    double ** A, ** localA, l, *msg;
    X=atoi(argv[1]);
    Y=X;

    //Extend dimension X with ghost cells if X%size!=0
    if (X%size!=0)
        X_ext=X+size-X%size;
    else
        X_ext=X;
      

    if (rank==0) {
        //Allocate and init matrix A
        A=malloc2D(X_ext,Y);
        init2D(A,X,Y);
    }
    //Local dimensions x,y
    x=X_ext/size;
    y=Y;

    //Allocate local matrix and scatter global matrix
    localA=malloc2D(x,y);
    double * idx;
    for (i=0;i<x;i++) {
        if (rank==0)
            idx=&A[i*size][0];
        MPI_Scatter(idx,Y,MPI_DOUBLE,&localA[i][0],y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    if (rank==0)
        free2D(A,X_ext,Y);
 
    //Timers   
    struct timeval ts,tf,comps,compf,comms,commf;
    double total_time=0,computation_time=0,communication_time=0;

	MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&ts,NULL);        
	msg = malloc(y * sizeof(double));
	
    for(k = 0; k < X - 1; k++){
		// if is owner_of_pivot_line(k) - x*rank <= k < x*(rank+1)
		if ( rank == (k % size) ) {
			//pack_data(lA, send_buffer);
			memcpy(msg, localA[ k/size ], y * sizeof(double) );
			//send_data_to_all
			gettimeofday(&comms,NULL);
			MPI_Bcast(msg,y,MPI_DOUBLE,rank,MPI_COMM_WORLD);
			gettimeofday(&commf,NULL);
			communication_time+=commf.tv_sec-comms.tv_sec+(commf.tv_usec-comms.tv_usec)*0.000001;
		}
		else {
			//receive_data_from_owner
			//unpack_data(receive_buffer, lA);
			srank = k % size;
			gettimeofday(&comms,NULL);
			MPI_Bcast(msg,y,MPI_DOUBLE,srank,MPI_COMM_WORLD);
			gettimeofday(&commf,NULL);
			communication_time+=commf.tv_sec-comms.tv_sec+(commf.tv_usec-comms.tv_usec)*0.000001;
		}
		//compute(k, lA);
		gettimeofday(&comps,NULL);
		
		srank = k % size;
		i = (k / size) + 1;
		for ( j = srank + 1; j < size; j++ ) {
			if (rank == j)
				i = k / size;
		}
		
		for ( i; i < x; i++ ) {
			l = localA[i][k] / msg[k];
			for ( j=k; j<y; j++ )		
				localA[i][j] -= l * msg[j];
		}
			
		gettimeofday(&compf,NULL);
		computation_time+=compf.tv_sec-comps.tv_sec+(compf.tv_usec-comps.tv_usec)*0.000001;
	}
	
	free(msg);
	
	MPI_Barrier(MPI_COMM_WORLD);
//********************************************************************************

    gettimeofday(&tf,NULL);
    total_time=tf.tv_sec-ts.tv_sec+(tf.tv_usec-ts.tv_usec)*0.000001;


    //Gather local matrices back to the global matrix
    if (rank==0) 
        A=malloc2D(X_ext,Y);
    for (i=0;i<x;i++) {
        if (rank==0)
            idx=&A[i*size][0];
        MPI_Gather(&localA[i][0],y,MPI_DOUBLE,idx,Y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    double avg_total,avg_comp,avg_comm,max_total,max_comp,max_comm;
    MPI_Reduce(&total_time,&max_total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&max_comp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&total_time,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&avg_comp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&avg_comm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    avg_total/=size;
    avg_comp/=size;
    avg_comm/=size;

    if (rank==0) {
        printf("LU-Cyclic-bcast\tSize\t%d\tProcesses\t%d\n",X,size);
        printf("Max times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",max_total,max_comp,max_comm);
        printf("Avg times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",avg_total,avg_comp,avg_comm);
    }

    //Print triangular matrix U to file
    if (rank==0) {
        char * filename="output_cyclic_bcast";
        print2DFile(A,X,Y,filename);
    }


    MPI_Finalize();

    return 0;
}


