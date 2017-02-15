#include "stdafx.h"

#include <math.h>

#include "cl_spd_inv.h"
#include "cl_common.h"


void get_delta_beta(cl_command_queue queue, cl_kernel kern_delta_beta, cl_mem matBuf,cl_mem diagAuxBuf,
	int matsize, dtype *delta, dtype *beta)
//void get_delta_beta(int matsize, SPDInv_structPtr SPDInvPtr, dtype *delta, dtype *beta)
{
	int argNum, excludeDiag, outOffset, inOffset;
	cl_int err;
	dtype res[2];
	//*********************step 1 得到每一行非对角元最大值，存到buf_diagBlk第一行， 同时将对角元素存到第二行
	argNum = 0;
	excludeDiag = 1;
	outOffset = 0;
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &diagAuxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &outOffset);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &matsize);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &excludeDiag);
	checkErr(err, __FILE__, __LINE__);

	size_t row_size = matsize;
	err = clEnqueueNDRangeKernel(queue, kern_delta_beta,
		1, NULL,	//1D work space
		&row_size,
		NULL,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	//err = clEnqueueReadBuffer(SPDInvPtr->queue, SPDInvPtr->buf_diagBlk, CL_TRUE, 0,
	//	sizeof(dtype) * 6, &out, 0, NULL, NULL);

	argNum = 0;
	excludeDiag = 0;
	outOffset =2*matsize;
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &diagAuxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &diagAuxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &outOffset);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &matsize);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &excludeDiag);

	row_size = 2;
	err = clEnqueueNDRangeKernel(queue, kern_delta_beta,
		1, NULL,	//1D work space
		&row_size,
		NULL,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	err = clEnqueueReadBuffer(queue, diagAuxBuf, CL_TRUE, sizeof(dtype)*outOffset,
		sizeof(dtype) * 2, &res, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	*delta = 1e-15*fmax(res[0] + res[1], 1);		//1e-15为double的有效精度
	*beta = fmax(res[1], 1e-15);
	*beta = fmax(*beta, res[0] / sqrt(matsize*matsize - 1));
	*beta = sqrt(*beta);

	//*****************************************************
}



/*
修正cholesky分解（modified cholesky）：
集成了对非正定的修正,enqueue on device
输入：
matBuf:待分解方阵,尺寸为matSize*matSize 
diagAuxBuf: >=2*matSize
输出：
matBuf：其下三角矩阵为分解的结果。
outMat：当copyOut为True时，结果拷贝到outMat
*/
//dtype cholesky_mod(int matsize, SPDInv_structPtr SPDInvPtr, dtype *outMat, bool copyOut)
void cholesky_mod(cl_command_queue queue, cl_kernel kern_cholesky_mod, cl_kernel kern_mat_max,
	cl_mem matBuf, cl_mem auxBuf,cl_mem diagBuf, cl_mem retBuf,
	int matSize, dtype *outMat, bool copyOut)
{
	int argNum;
	cl_int err;
	dtype beta, delta, ret;
	dtype *debug;
	dtype data[2*18];

	//get delta,beta
	get_delta_beta(queue, kern_mat_max, matBuf, auxBuf, matSize, &delta, &beta);

	//set kernel arguments
	argNum = 0;
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(cl_mem), &auxBuf);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(cl_mem), &diagBuf);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(cl_mem), &retBuf);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(int), &matSize);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(dtype), &delta);
	err = clSetKernelArg(kern_cholesky_mod, argNum++, sizeof(dtype), &beta);
	checkErr(err, __FILE__, __LINE__);

	//for (int j = 0; j < matSize; j++) {
		int j = 0;
		err = clSetKernelArg(kern_cholesky_mod, argNum, sizeof(int), &j);		//块矩阵的第j列
		//执行内核
		size_t global_size[2] = { matSize-j,1 };
		err = clEnqueueNDRangeKernel(queue, kern_cholesky_mod,
			1, NULL,	//1D work space
			global_size,
			global_size,
			0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
		clFinish(queue);

		err = clEnqueueReadBuffer(queue, auxBuf, CL_TRUE, 0,
			sizeof(dtype)*2*18, data, 0, NULL, NULL);
		printf("col %d diag:\n", j);
		for(int k = 0;k < matSize;k++)
			printf("%le  ", data[k]);
		printf("\n");

	//}
	err = clEnqueueReadBuffer(queue, retBuf, CL_TRUE, 0,
			sizeof(dtype), &ret, 0, NULL, NULL);

	// 读取结果
	if (copyOut)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*matSize*matSize, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_CHOLESKY==1
	if (copyOut)		//
		debug = outMat;
	else {
		debug = (dtype*)malloc(sizeof(dtype)*matSize*matSize);
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*matSize*matSize, debug, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

	printf("L(cholesky):\n");
	for (int r = 0; r < matSize; r++)
	{
		for (int c = 0; c < matSize; c++)
			printf("%le\t", debug[r*matSize + c]);
		printf("\n");
	}
	if (!copyOut)		//
		free(debug);

#endif
}


/*
计算modified cholesky修正的对角元素E
输入：
	buf_diag --原矩阵的对角元素
输出：
	buf_diag --修正的对角元素E
*/
void compute_cholmod_E(cl_command_queue queue, cl_kernel kern_cholmod_E,
	cl_mem matBuf,cl_mem buf_diag, int matSize, dtype *Eout)
{
	cl_int err;
	//set kernel arguments
	int argNum = 0;
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(cl_mem), &buf_diag);
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(int), &matSize);

	//执行内核
	size_t global_size[1] = { matSize};
	err = clEnqueueNDRangeKernel(queue, kern_cholmod_E,
		1, NULL,	//1D work space
		global_size,
		global_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	if(Eout!=NULL)
	{
		err = clEnqueueReadBuffer(queue, buf_diag, CL_TRUE, 0,
			sizeof(dtype)*matSize, Eout, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}
}