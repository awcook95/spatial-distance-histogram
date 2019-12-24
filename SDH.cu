/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

/* USF Fall 2019 CIS4930 Programming on Massively Parallel Systems
   Project Description: Write a CUDA program to implement the same
   functionality as the CPU only code

   Student: Alexander Cook
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

bucket * histogram;		/* list of all buckets in the histogram   */
bucket * z_histogram;   /* histogram initialized to all 0s        */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	Distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

//Device helper function which now takes a pointer as an argument instead of using a global pointer
__device__ double d_p2p_distance(atom *atom_list, int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__global__ void PDH_kernel(atom *d_atom_list, bucket *d_histogram, int PDH_acnt, int PDH_res){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int h_pos;
	double dist;

	if(i < j && i < PDH_acnt && j < PDH_acnt){ // i < j so distances are not counted twice
		dist = d_p2p_distance(d_atom_list, i,j);
			h_pos = (int) (dist / PDH_res);
			atomicAdd(&(d_histogram[h_pos].d_cnt), 1);
			 
	}
	
}

//Single threaded kernel for testing
__global__ void PDH_kernelST(atom *d_atom_list, bucket *d_histogram, int PDH_acnt, int PDH_res){
	int i = threadIdx.x;

	int j, h_pos;
	double dist;
	
	for(; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = d_p2p_distance(d_atom_list,i,j);
			h_pos = (int) (dist / PDH_res);
			d_histogram[h_pos].d_cnt++;
		} 
	}
	
}


/* 
	Brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	Set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(const char* version) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for %s version: %ld.%06ld\n",version , sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	Print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

//Prints difference between two histograms
void output_histogram_diff(bucket *histo1, bucket *histo2){
	int i; 
	long long total_cnt1 = 0;
	long long total_cnt2 = 0;
	printf("Difference between CPU and GPU histogram\n");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histo1[i].d_cnt - histo2[i].d_cnt);
		total_cnt1 += histo1[i].d_cnt;
		total_cnt2 += histo2[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt1 - total_cnt2);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);

	//Allocate host memory
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	z_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	//initialize histogram to zero
	memset(z_histogram, 0, sizeof(bucket)*num_buckets);

	//Allocate device memory
	bucket *d_histogram; //pointer to array of buckets
	atom *d_atom_list; //pointer to array of atoms

	cudaMalloc((void**)&d_histogram, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* Generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	//Copy host data to device memory
	cudaMemcpy(d_histogram, z_histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	
	/* Start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* Call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* Report running time for CPU version */ 
	report_running_time("CPU");
	
	/* Print out the histogram */
	output_histogram(histogram);


	//Define 2D block and grid size
	int num_threads = 16; //number of threads in one dimension of a block
	dim3 blockDim(num_threads,num_threads); //num_threads^2 threads per block
	int num_blocks = (PDH_acnt + num_threads - 1)/num_threads; //calculate number of blocks for the grid in a particular dimension
	dim3 gridDim(num_blocks, num_blocks); //the grid is the same size in x and y dimension
	
	//Start counting time
	gettimeofday(&startTime, &Idunno);

	//Launch kernel
	PDH_kernel<<<gridDim,blockDim>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);
	//PDH_kernelST<<<1,1>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);

	//Copy data from gpu memory to host memory
	bucket * GPU_histogram;
	GPU_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	cudaMemcpy(GPU_histogram, d_histogram, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	
	//Report GPU running time
	report_running_time("GPU");

	/* Print out the histogram again for gpu version */
	output_histogram(GPU_histogram);

	//print difference between the two histograms
	output_histogram_diff(histogram, GPU_histogram);

	free(histogram);
	free(atom_list);
	free(GPU_histogram);
	cudaFree(d_histogram);
	cudaFree(d_atom_list);
	

	return 0;
}


