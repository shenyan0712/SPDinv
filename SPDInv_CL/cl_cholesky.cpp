#include "stdafx.h"

#include <math.h>

#include "cl_spd_inv.h"
#include "cl_common.h"

/*
cholesky分解：方法1
特性：
分块并行方法，enqueue on device
输入：
matBuf--待分解方阵, matSize*matSize
diagAuxBuf--辅助Buffer, >=3*matSize
输出：
matBuf--其下三角存放分解结果
diagAuxBuf--存放Ljj^-1子块
outMat!=NULL 则将分解结果拷贝到outMat
ret= --> 0.0 good 1.0 not good
*/
dtype cholesky_m1(cl_command_queue queue, cl_kernel kern_cholesky,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf,
	int matSize, dtype *outMat)
{
	cl_int err;
	dtype ret;

	//set kernel arguments
	err = clSetKernelArg(kern_cholesky, 0, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholesky, 1, sizeof(cl_mem), &diagAuxBuf);
	err = clSetKernelArg(kern_cholesky, 2, sizeof(cl_mem), &retBuf);
	err = clSetKernelArg(kern_cholesky, 3, sizeof(dtype) * 9, NULL);		//T_ii块
	err = clSetKernelArg(kern_cholesky, 4, sizeof(dtype) * 6, NULL);		//L_ii,只存储下三角
	err = clSetKernelArg(kern_cholesky, 5, sizeof(int), &matSize);

	//执行内核
	int j = 0;
	size_t global_size[2] = { matSize,3 };
	size_t local_size[2] = { 3,3 };
	err = clSetKernelArg(kern_cholesky, 6, sizeof(int), &j);		//块矩阵的第j列
	err = clEnqueueNDRangeKernel(queue, kern_cholesky,
		2, NULL,	//2D work space
		global_size,
		local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	err = clEnqueueReadBuffer(queue, retBuf, CL_TRUE, 0,
		sizeof(ret), &ret, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	// 读取结果
	if (outMat!=NULL)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*matSize*matSize, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_CHOLESKY==1
	printBuf2D(stdout, queue, matBuf, matSize, matSize, "L(cholesky):");
	printBuf2D(stdout, queue, diagAuxBuf, 3, matSize, "diagInv:");

#endif
	return ret;
}

