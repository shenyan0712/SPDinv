#include "stdafx.h"

#include <math.h>

#include "cl_spd_inv.h"
#include "cl_common.h"


/*
修正cholesky分解（modified cholesky）：
集成了对非正定的修正,enqueue on device
输入：
matBuf:待分解方阵,尺寸为matSize*matSize
auxBuf: >=2*matSize
diagBuf: 存放原始矩阵的对角元素
输出：
matBuf：其下三角矩阵为分解的结果。
outMat：当copyOut为True时，结果拷贝到outMat
*/
//dtype cholesky_mod(int matsize, SPDInv_structPtr SPDInvPtr, dtype *outMat, bool copyOut)
void cholmod_blk(cl_command_queue queue, cl_kernel kern_cholmod_blk, cl_kernel kern_mat_max,
	cl_mem matBuf, cl_mem auxBuf, cl_mem diagInvBuf, cl_mem diagBuf, cl_mem retBuf,
	int matSize, dtype *outMat, bool copyOut)
{
	int argNum;
	cl_int err;
	dtype beta, delta, ret;
	dtype *debug;
	dtype data[2 * 18];

	//get delta,beta
	get_delta_beta(queue, kern_mat_max, matBuf, auxBuf, matSize, &delta, &beta);

	//set kernel arguments
	argNum = 0;
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &auxBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &diagInvBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &diagBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &retBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype) * 9, NULL);		//T_ii块
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype) * 6, NULL);		//L_ii,只存储下三角
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(int), &matSize);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype), &beta);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype), &delta);
	checkErr(err, __FILE__, __LINE__);

	//for (int j = 0; j < matSize/3; j++) {
	int j = 0;
		err = clSetKernelArg(kern_cholmod_blk, argNum, sizeof(int), &j);		//块矩阵的第j列
																				//执行内核
		size_t global_size[2] = { 3,3 };
		size_t local_size[2] = { 3,3 };
		err = clEnqueueNDRangeKernel(queue, kern_cholmod_blk,
			2, NULL,	//1D work space
			global_size,
			local_size,
			0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
		clFinish(queue);

		printBuf2D(stdout, queue, matBuf, matSize, matSize, "******A:");
		printBuf2D(stdout, queue, diagInvBuf, 3, matSize, "******Linv:");
		printBuf2D(stdout, queue, auxBuf, 3, matSize, "******aux:");
		printBuf1D(stdout, queue, diagBuf, matSize, "******diag:");
		printBuf1D(stdout, queue, retBuf, 1, "******ret:");
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