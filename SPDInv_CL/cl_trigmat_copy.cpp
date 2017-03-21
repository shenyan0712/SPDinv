#include "stdafx.h"

#include "cl_common.h"
#include "cl_spd_inv.h"

/*
计算M_inv=(L^-t)*L^-1
buf_spd_A的上三角已经是L^-t

*/
void trigMat_copy(cl_command_queue queue, cl_kernel kern_trigMat_copy, cl_mem inBuf, cl_mem outBuf,
	int mat_size, dtype *outMat)
{
	cl_command_queue queue_device;
	cl_int err;
	dtype ret = 0.0;

	//********配置参数**********/
	int j = 0;
	err = clSetKernelArg(kern_trigMat_copy, 0, sizeof(cl_mem), &inBuf);
	err |= clSetKernelArg(kern_trigMat_copy, 1, sizeof(cl_mem), &outBuf);
	err |= clSetKernelArg(kern_trigMat_copy, 2, sizeof(int), &mat_size);

	//执行内核,
	size_t global_size[2] = { mat_size,mat_size};
	//size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(queue, kern_trigMat_copy,
		2, NULL,	//1D work space
		global_size,
		NULL,		//local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, inBuf, CL_TRUE, 0,
			sizeof(dtype)*mat_size*mat_size, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_TRIGMAT_INV==1
	printBuf2D(stdout, queue, matBuf, mat_size, mat_size,"Inverse:");

#endif

}