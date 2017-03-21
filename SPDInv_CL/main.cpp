// symPosDefMatInv.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream> 
#include <windows.h> 

#include <clBLAS.h>

#include "stdafx.h"
#include "cl_common.h"
#include "cl_spd_inv.h"
#include "spd_inv.h"


using namespace std;

#define NUM_ITER			20
#define NUM_TEST_PER_MAT	30
#define MAX_MAT_SIZE  10000
#define MAX_BLK_SIZE	4

time_t cur_tm, total_tm, tm_diff[NUM_TEST_PER_MAT];		//
time_t copy_tm, cur_tm2;
double av_tm;

SPDInv_struct spdInvStruct;
int blksize = 3;

void clBLAS_test();


void generate_spd(int size, dtype *spdMat)
{
	dtype *Lmat;
	dtype *ptr;
	int i, j, k;

	Lmat = (dtype*)malloc(sizeof(dtype)*size*size);
	memset(Lmat, 0, sizeof(dtype)*size*size);

	memset(spdMat, 0, sizeof(dtype)*size*size);

	//generate lower triangular matrix
	//srand((unsigned)time(NULL));
	for (i = 0;i < size;i++) {
		ptr = Lmat + i*size;
		for (j = 0;j <= i;j++) {
			*ptr = rand()/65536.0 + 1;
			ptr++;
		}
	}

	//display
#if DEBUG_GEN_MAT==1
	printf("lower triangular matrix:\n");
	for (i = 0;i < size;i++) {
		ptr = Lmat + i*size;
		for (j = 0;j < size;j++) {
			printf("%f  ",*ptr);
			ptr++;
		}
		printf("\n");
	}
#endif

	//matrix multiplication
	dtype *ptr_i, *ptr_j, *ptr_elem;
	dtype sum;
	for (i = 0; i < size; i++) {
		ptr_i = Lmat + i*size;		//ith row
		for (j = 0;j < size;j++) {
			ptr_j= Lmat + j*size;		//ith row
			ptr_elem = spdMat + i*size+j;
			sum = 0;
			for (k = 0;k < size;k++) {
				sum += (*(ptr_i+k))*(*(ptr_j+k));
			}
			if(i==j)
				*ptr_elem = sum+100;
			else
				*ptr_elem = sum;
		}
	}

#if DEBUG_GEN_MAT==1
	//display for spd matrix
	printf("spd matrix:\n");
	for (i = 0;i < size;i++) {
		ptr = spdMat + i*size;
		for (j = 0;j < size;j++) {
			printf("%f  ", *ptr);
			ptr++;
		}
		printf("\n");
	}
#endif

	free(Lmat);
}

void mod_cholesky_test();
void cholesky_m1_test();

int main()
{
	dtype ret;
	clblasStatus stat;
	cl_int err;
	int matsize;

	dtype *spdMat, *outMat, *bakMat;

	//clBLAS_test();
	//mod_cholesky_test();
	//system("pause");
	//return 0;


	outMat = (dtype *)malloc(sizeof(dtype)*MAX_MAT_SIZE*MAX_MAT_SIZE);
	spdMat = (dtype *)malloc(sizeof(dtype)*MAX_MAT_SIZE*MAX_MAT_SIZE);
	//bakMat = (dtype *)malloc(sizeof(dtype)*MAX_MAT_SIZE*MAX_MAT_SIZE);


	cl_SPDInv_setup(&spdInvStruct, MAX_MAT_SIZE, MAX_BLK_SIZE);
	err = clblasSetup();


	/**************************************/
	/*************test********************/
	for (int i = 1; i <= NUM_ITER; i++)
	{
		copy_tm = 0.0;
		total_tm = 0.0;
		matsize = i * 120;		//120
		//生成NxN的对称正定矩阵
		//generate_spd(matsize, spdMat);

		gen_rand(spdInvStruct.queue, spdInvStruct.kern_gen_rand, spdInvStruct.buf_spd_B, matsize, NULL);
		clEnqueueCopyBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_B, spdInvStruct.buf_spd_A, 0, 0, sizeof(dtype)*matsize*matsize, 0, NULL, NULL);
		//printBuf2D(stdout, spdInvStruct.queue, spdInvStruct.buf_spd_B, matsize, matsize, "origin:");
		stat = clblasDtrmm(clblasRowMajor, clblasRight, clblasLower,	// B<--A*B, A是上三角, B是下三角
			clblasTrans, clblasNonUnit, matsize, matsize,
			1, spdInvStruct.buf_spd_B, 0, matsize,
			spdInvStruct.buf_spd_A, 0, matsize,
			1, &spdInvStruct.queue, 0, NULL, NULL);
		clEnqueueReadBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0, sizeof(dtype)*matsize*matsize, spdMat, 0, NULL, NULL);

		/*
		//***********************test cholesky***********************************
		//*****warm up
		ret = cholesky_m1(spdInvStruct.queue, spdInvStruct.kern_cholesky_m1,
			spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux, spdInvStruct.buf_ret,
			matsize, NULL);
		if (ret == 1.0)
		{
		printf("chol failed.\n");
		system("pause");
		exit(1);
		}
		//*****
		cur_tm = clock();
		for (int j = 0; j < NUM_TEST_PER_MAT; j++) {
			//copy data to buffer
			cur_tm2 = clock();
			err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
				sizeof(dtype)*matsize*matsize, spdMat, 0, NULL, NULL);
			clFlush(spdInvStruct.queue);
			copy_tm = (clock() - cur_tm2);
			ret = cholesky_m1(spdInvStruct.queue, spdInvStruct.kern_cholesky_m1,
				spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux, spdInvStruct.buf_ret,
				matsize, outMat);
		}
		total_tm = clock() - cur_tm;
		av_tm = ((double)(total_tm-copy_tm)) / NUM_TEST_PER_MAT;
		printf("mat N=%d, av.tm=%lf\n",matsize, av_tm);
		*/

		/*
		//***************************************test trigInv****************************************
		//*****warm up
		ret = cholesky_m1(spdInvStruct.queue,spdInvStruct.kern_cholesky_m1,spdInvStruct.buf_spd_A,spdInvStruct.buf_diagAux,
			spdInvStruct.buf_ret, matsize, outMat);
		if (ret != 0.0)
		{
			printf("chol failed.\n");
			system("pause");
			exit(1);
		}
		//*****
		cur_tm = clock();
		for (int j = 0; j < NUM_TEST_PER_MAT; j++) {
			//copy data to buffer
			cur_tm2 = clock();
			err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
				sizeof(dtype)*matsize*matsize, outMat, 0, NULL, NULL);
			clFlush(spdInvStruct.queue);
			checkErr(err, __FILE__, __LINE__);
			copy_tm += (clock() - cur_tm2);
			trigMat_inv_m1(spdInvStruct.queue,spdInvStruct.kern_trigMat_inv_m1, spdInvStruct.buf_spd_A,spdInvStruct.buf_diagAux,
			  spdInvStruct.buf_ret,	matsize, NULL);
		}
		total_tm = clock() - cur_tm;
		av_tm = ((double)(total_tm- copy_tm)) / NUM_TEST_PER_MAT;
		printf("mat N=%d, av.tm=%lf, copy_tm=%f\n", matsize, av_tm, (double)copy_tm/NUM_TEST_PER_MAT);
		*/

		///*
		//**************************************test SPDInv*********************************
		//*****warm up
		//err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
		//	sizeof(dtype)*matsize*matsize, spdMat, 0, NULL, NULL);
		//clFlush(spdInvStruct.queue);
		//printBuf2D(stdout, spdInvStruct.queue, spdInvStruct.buf_spd_A, matsize, matsize, "rand:");
		ret=cholesky_m1(spdInvStruct.queue, spdInvStruct.kern_cholesky_m1,spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux,
			spdInvStruct.buf_ret, matsize, NULL);
		if (ret != 0.0)
		{
			printf("chol failed.\n");
			system("pause");
			exit(1);
		}
		//
		cur_tm = clock();
		for (int j = 0; j < NUM_TEST_PER_MAT; j++) {
			//copy data to buffer
			cur_tm2 = clock();
			err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
				sizeof(dtype)*matsize*matsize, spdMat, 0, NULL, NULL);
			clFlush(spdInvStruct.queue);
			checkErr(err, __FILE__, __LINE__);
			copy_tm += (clock() - cur_tm2);

			ret=cholesky_m1(spdInvStruct.queue, spdInvStruct.kern_cholesky_m1, spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux,
				spdInvStruct.buf_ret, matsize, NULL);
			if (ret != 0.0)
			{
				printf("chol failed.\n");
				system("pause");
				exit(1);
			}
			trigMat_inv_m1(spdInvStruct.queue, spdInvStruct.kern_trigMat_inv_m1, spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux,
				spdInvStruct.buf_ret, matsize, NULL);
			
			//used for CPU
			//trigMat_mul(spdInvStruct.queue, spdInvStruct.kern_trigMat_mul, spdInvStruct.buf_spd_A, spdInvStruct.buf_diagAux, spdInvStruct.buf_spd_B,
			//	matsize, NULL);

			///*
			//used for AMD GPU
			trigMat_copy(spdInvStruct.queue, spdInvStruct.kern_trigMat_copy, spdInvStruct.buf_spd_A, spdInvStruct.buf_spd_B, matsize, NULL);
			stat=clblasDtrmm(clblasRowMajor, clblasRight, clblasUpper,	// B<--A*B, A是上三角, B是下三角
				clblasNoTrans, clblasDiag::clblasNonUnit, 4, 4,
				1, spdInvStruct.buf_spd_A, 0, 4,
				spdInvStruct.buf_spd_B, 0, 4,
				1, &spdInvStruct.queue, 0, NULL, NULL);
			if (stat != clblasStatus::clblasSuccess)
			{
				printf("trig mul failed.\n"); system("pause");
				exit(1);
			}
			//*/
		}
		total_tm = clock() - cur_tm;
		av_tm = ((double)total_tm- copy_tm) / NUM_TEST_PER_MAT;
		printf("mat N=%d, av.tm=%f, copy_tm=%f\n", matsize, av_tm, (dtype)copy_tm/ NUM_TEST_PER_MAT);
		//*/
	}

	system("pause");

	free(spdMat);
	free(outMat);

    return 0;
}

extern dtype A[30][30];
void mod_cholesky_test()
{
	cl_int err;
	int matsize =30;
	//dtype mat[3][3] = { {4,2,1},{2,6,6},{1,6,5} };
	//dtype out[3][3];
	/*
	dtype mat[4][4] = { {1,     3 ,    1 ,    2},
						{3,     12,    13 ,   22},
					{1 ,   13 ,   35  ,  51},
						{2  ,  22 ,   51 ,   93} };
	*/
	/*
	dtype mat[9][9] = { {1.000000 , 1.000000,  3.000000,  5.000000 , 1.000000,  2.000000 , 4.000000 , 3.000000 , 4.000000},
	{1.000000 , 0.500000 , 7.000000,  9.000000,  9.000000 , 4.000000 , 8.000000,  9.000000,  50.000000},
	{3.000000,  7.000000,  29.000000,  23.000000,  39.000000 , 24.000000 , 60.000000,  23.000000 , 66.000000 },
	{5.000000,  9.000000,  23.000000 , 39.000000,  26.000000,  40.000000 , 47.000000,  26.000000,  71.000000},
	{1.000000,  9.000000,  39.000000,  26.000000,  71.000000,  53.000000 , 103.000000,  36.000000,  119.000000},
	{2.000000,  4.000000 , 24.000000,  40.000000,  53.000000 , 103.000000,  113.000000,  43.000000,  80.000000},
	{4.000000,  8.000000,  60.000000,  47.000000,  103.000000,  113.000000,  203.000000,  79.000000,  148.000000},
	{3.000000,  9.000000,  23.000000,  26.000000,  36.000000,  43.000000,  79.000000,  62.000000,  143.000000},
	{4.000000,  50.000000,  66.000000,  71.000000,  119.000000,  80.000000,  148.000000,  143.000000,  719.000000} };
	*/

	///*
	dtype mat[18][18] = {
		{-1.147157e+09, -1.426342e+09, 4.725405e+08, -5.568353e+10, 2.628552e+10, -3.342550e+09, -1.411506e+09, -1.832157e+09, 6.235531e+08, -7.474564e+10, 3.059464e+10, 1.417797e+09, 3.054214e+07, -6.912809e+07, 2.056894e+07, -3.919537e+09, -1.702177e+09, 9.931234e+07},
		{-1.426342e+09, 8.331292e+06, 1.174761e+07, 1.174655e+10, 5.615483e+10, 1.929104e+09, -1.913704e+09, 1.921373e+08, -6.657815e+07, 2.481519e+10, 7.642410e+10, -8.695181e+08, -5.674843e+07, 1.193685e+08, -3.691627e+07, 6.779897e+09, 3.159746e+09, -2.051712e+08 },
		{4.725405e+08, 1.174761e+07, -7.893033e+06, -3.211985e+09, -1.845112e+10, -5.646933e+08, 6.314626e+08, -4.544853e+07, 1.577805e+07, -7.311552e+09, -2.497823e+10, 2.897258e+08, 1.363377e+07, -2.989371e+07, 9.043460e+06, -1.696211e+09, -7.595233e+08, 4.652669e+07},
		{-5.568353e+10, 1.174655e+10, -3.211985e+09, 9.788477e+11, 2.341609e+12, 1.154932e+11, -7.579814e+10, 2.338145e+10, -7.983287e+09, 1.726212e+12, 3.231274e+12, -5.038497e+10, -2.989894e+09, 6.296930e+09, -1.946117e+09, 3.576425e+11, 1.664795e+11, -1.079211e+10},
		{2.628552e+10, 5.615483e+10, -1.845112e+10, 2.341609e+12, -2.909189e+11, 1.553820e+11, 3.031899e+10, 7.463615e+10, -2.541118e+10, 3.266912e+12, -1.968657e+11, -6.849157e+10, -1.702853e+09, 3.844853e+09, -1.145467e+09, 2.180138e+11, 9.490048e+10, -5.558309e+09},
		{-3.342550e+09, 1.929104e+09, -5.646933e+08, 1.154932e+11, 1.553820e+11, 1.228732e+10, -4.716219e+09, 3.111805e+09, -1.038957e+09, 1.859419e+11, 2.213759e+11, -4.106882e+09, -4.220173e+08, 8.587171e+08, -2.703755e+08, 4.881382e+10, 2.348842e+10, -1.591724e+09},
		{-1.411506e+09, -1.913704e+09, 6.314626e+08, -7.579814e+10, 3.031899e+10, -4.716219e+09, -1.696111e+09, -2.475834e+09, 8.444759e+08, -1.026345e+11, 3.308456e+10, 2.078532e+09, 4.340130e+07, -9.858169e+07, 2.927909e+07, -5.589093e+09, -2.418958e+09, 1.403328e+08},
		{-1.832157e+09, 1.921373e+08, -4.544853e+07, 2.338145e+10, 7.463615e+10, 3.111805e+09, -2.475834e+09, 5.277479e+08, -1.774406e+08, 4.530071e+10, 1.022950e+11, -1.341126e+09, -8.488157e+07, 1.809997e+08, -5.556959e+07, 1.027701e+10, 4.726997e+09, -3.013017e+08},
		{6.235531e+08, -6.657815e+07, 1.577805e+07, -7.983287e+09, -2.541118e+10, -1.038957e+09, 8.444759e+08, -1.774406e+08, 6.096646e+07, -1.527062e+10, -3.493343e+10, 5.140337e+08, 2.236883e+07, -4.929799e+07, 1.487363e+07, -2.796901e+09, -1.246227e+09, 7.576364e+07},
		{-7.474564e+10, 2.481519e+10, -7.311552e+09, 1.726212e+12, 3.266912e+12, 1.859419e+11, -1.026345e+11, 4.530071e+10, -1.527062e+10, 2.983182e+12, 4.542517e+12, -7.821117e+10, -4.606015e+09, 9.832949e+09, -3.017032e+09, 5.582911e+11, 2.565095e+11, -1.632441e+10},
		{3.059464e+10, 7.642410e+10, -2.497823e+10, 3.231274e+12, -1.968657e+11, 2.213759e+11, 3.308456e+10, 1.022950e+11, -3.493343e+10, 4.542517e+12, 6.306019e+10, -1.010110e+11, -2.405386e+09, 5.454173e+09, -1.621354e+09, 3.092367e+11, 1.340604e+11, -7.798949e+09},
		{1.417797e+09, -8.695181e+08, 2.897258e+08, -5.038497e+10, -6.849157e+10, -4.106882e+09, 2.078532e+09, -1.341126e+09, 5.140337e+08, -7.821117e+10, -1.010110e+11, 4.400334e+09, -1.586479e+08, 2.934610e+08, -9.743096e+07, 1.672408e+10, 8.820361e+09, -6.651589e+08},
		{3.054214e+07, -5.674843e+07, 1.363377e+07, -2.989894e+09, -1.702853e+09, -4.220173e+08, 4.340130e+07, -8.488157e+07, 2.236883e+07, -4.606015e+09, -2.405386e+09, -1.586479e+08, 3.067776e+07, -6.351859e+07, 1.980986e+07, -3.609125e+09, -1.707527e+09, 1.131721e+08},
		{-6.912809e+07, 1.193685e+08, -2.989371e+07, 6.296930e+09, 3.844853e+09, 8.587171e+08, -9.858169e+07, 1.809997e+08, -4.929799e+07, 9.832949e+09, 5.454173e+09, 2.934610e+08, -6.351859e+07, 1.410164e+08, -4.238219e+07, 7.998860e+09, 3.539124e+09, -2.128073e+08},
		{2.056894e+07, -3.691627e+07, 9.043460e+06, -1.946117e+09, -1.145467e+09, -2.703755e+08, 2.927909e+07, -5.556959e+07, 1.487363e+07, -3.017032e+09, -1.621354e+09, -9.743096e+07, 1.980986e+07, -4.238219e+07, 1.299408e+07, -2.406232e+09, -1.103243e+09, 6.999987e+07},
		{-3.919537e+09, 6.779897e+09, -1.696211e+09, 3.576425e+11, 2.180138e+11, 4.881382e+10, -5.589093e+09, 1.027701e+10, -2.796901e+09, 5.582911e+11, 3.092367e+11, 1.672408e+10, -3.609125e+09, 7.998860e+09, -2.406232e+09, 4.537523e+11, 2.010886e+11, -1.212221e+10},
		{-1.702177e+09, 3.159746e+09, -7.595233e+08, 1.664795e+11, 9.490048e+10, 2.348842e+10, -2.418958e+09, 4.726997e+09, -1.246227e+09, 2.565095e+11, 1.340604e+11, 8.820361e+09, -1.707527e+09, 3.539124e+09, -1.103243e+09, 2.010886e+11, 9.505777e+10, -6.293168e+09},
		{9.931234e+07, -2.051712e+08, 4.652669e+07, -1.079211e+10, -5.558309e+09, -1.591724e+09, 1.403328e+08, -3.013017e+08, 7.576364e+07, -1.632441e+10, -7.798949e+09, -6.651589e+08, 1.131721e+08, -2.128073e+08, 6.999987e+07, -1.212221e+10, -6.293168e+09, 4.666134e+08} };
	mat[0][0] = 100;
	//*/

	dtype out[30][30];
	dtype diag[30];

	cl_SPDInv_setup(&spdInvStruct, MAX_MAT_SIZE, MAX_BLK_SIZE);

	err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
		sizeof(dtype)*matsize*matsize,(void*)A, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFlush(spdInvStruct.queue);

	//cholesky_mod(spdInvStruct.queue,spdInvStruct.kern_cholesky_mod,spdInvStruct.kern_mat_max,
	//	spdInvStruct.buf_spd_A,spdInvStruct.buf_aux,spdInvStruct.buf_diag, spdInvStruct.buf_ret,
	//	matsize, (dtype*)out, true);

	cholmod_blk(spdInvStruct.queue, spdInvStruct.kern_cholmod_blk, spdInvStruct.kern_mat_max,
		spdInvStruct.buf_spd_A, spdInvStruct.buf_blkBackup, spdInvStruct.buf_diagAux, spdInvStruct.buf_diag,  spdInvStruct.buf_ret,
		matsize, (dtype*)out);

	compute_cholmod_E(spdInvStruct.queue, spdInvStruct.kern_cholmod_E,
		spdInvStruct.buf_spd_A, spdInvStruct.buf_diag, matsize, (dtype*)diag);

	printf("E:\n");
	for (int c = 0; c < matsize; c++)
		printf("%le\t", diag[c]);
	printf("\n");

}


void clBLAS_test()
{
	cl_int err;
	size_t lda = 4;
	dtype A[4 * 4] = {
		1,2,3,4,
		0,6,7,8,
		0,0,11,12,
		0,0,0,16,
	};

	dtype B[4 * 4] = {
		1,2,3,4,
		5,6,7,8,
		9,10,11,12,
		13,14,15,16,
	};

	dtype out[4 * 4];

	cl_SPDInv_setup(&spdInvStruct, MAX_MAT_SIZE, MAX_BLK_SIZE);
	clblasSetup();

	err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, CL_TRUE, 0,
		sizeof(dtype)*4*4, (void*)A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_B, CL_TRUE, 0,
		sizeof(dtype) * 4 * 4, (void*)B, 0, NULL, NULL);
	//err = clEnqueueCopyBuffer(spdInvStruct.queue, spdInvStruct.buf_spd_A, spdInvStruct.buf_spd_B,0,0, sizeof(dtype) * 4 * 4,0,NULL,NULL);

	checkErr(err, __FILE__, __LINE__);
	clFlush(spdInvStruct.queue);
	printBuf2D(stdout, spdInvStruct.queue, spdInvStruct.buf_spd_A, 4, 4, "A:");

	/*
	clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
		4, 4, 4,
		1,spdInvStruct.buf_spd_A, 0, lda,
		spdInvStruct.buf_spd_A, 0, lda,
		1,spdInvStruct.buf_spd_B, 0, lda,
		1, &spdInvStruct.queue, 0, NULL, NULL);
	*/
	clblasDtrmm(clblasRowMajor, clblasRight, clblasUpper,
		clblasTrans, clblasDiag::clblasNonUnit, 4, 4,
		1, spdInvStruct.buf_spd_B, 0, 4,
		spdInvStruct.buf_spd_A, 0, 4,
		1, &spdInvStruct.queue, 0, NULL, NULL);



	printBuf2D(stdout, spdInvStruct.queue, spdInvStruct.buf_spd_A, 4, 4, "out:");



}