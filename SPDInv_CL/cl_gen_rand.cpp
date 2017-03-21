#include "stdafx.h"

#include <time.h>

#include "cl_common.h"
#include "cl_spd_inv.h"

/*
计算M_inv=(L^-t)*L^-1
buf_spd_A的上三角已经是L^-t

*/
void gen_rand(cl_command_queue queue, cl_kernel kern_gen_rand,cl_mem outBuf,
	int mat_size, dtype *outMat)
{
	dtype seed;
	cl_command_queue queue_device;
	cl_int err;
	dtype ret = 0.0;

	srand((unsigned)time(NULL));
	seed = rand() / 6553.60 + 1;


	//********配置参数**********/
	int j = 0;
	err = clSetKernelArg(kern_gen_rand, 0, sizeof(cl_mem), &outBuf);
	err |= clSetKernelArg(kern_gen_rand, 1, sizeof(int), &mat_size);
	err |= clSetKernelArg(kern_gen_rand, 2, sizeof(dtype), &seed);

	//执行内核,
	size_t global_size[2] = { mat_size,mat_size};
	//size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(queue, kern_gen_rand,
		2, NULL,	//1D work space
		global_size,
		NULL,		//local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, outBuf, CL_TRUE, 0,
			sizeof(dtype)*mat_size*mat_size, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_TRIGMAT_INV==1
	printBuf2D(stdout, queue, matBuf, mat_size, mat_size,"Inverse:");

#endif

}