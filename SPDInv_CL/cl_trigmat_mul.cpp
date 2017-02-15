#include "stdafx.h"

#include "cl_common.h"
#include "cl_spd_inv.h"

/*
计算M_inv=(L^-t)*L^-1
buf_spd_A的上三角已经是L^-t

*/
void trigMat_mul(int mat_size, SPDInv_structPtr SPDInvPtr, dtype *outMat)
{
	cl_command_queue queue_device;
	cl_int err;
	dtype ret = 0.0;

	//********配置参数**********/
	int j = 0;
	err = clSetKernelArg(SPDInvPtr->kern_trigMat_mul, 0, sizeof(cl_mem), &SPDInvPtr->buf_spd_A);
	err |= clSetKernelArg(SPDInvPtr->kern_trigMat_mul, 1, sizeof(cl_mem), &SPDInvPtr->buf_aux);
	err |= clSetKernelArg(SPDInvPtr->kern_trigMat_mul, 2, sizeof(int), &mat_size);

	//执行内核,
	size_t global_size[2] = { mat_size,mat_size};
	//size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(SPDInvPtr->queue, SPDInvPtr->kern_trigMat_mul,
		2, NULL,	//1D work space
		global_size,
		NULL,		//local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(SPDInvPtr->queue);

#if DEBUG_TRIGMAT_INV==1
	err = clEnqueueReadBuffer(SPDInvPtr->queue, SPDInvPtr->buf_spd_A, CL_TRUE, 0,
		sizeof(dtype)*mat_size*mat_size, outMat, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	//display for check
	printf("spd inv:\n");
	for (int r = 0; r < mat_size; r++)
	{
		for (int c = 0; c < mat_size; c++)
			printf("%le\t", outMat[r*mat_size + c]);
		printf("\n");
	}
#endif

}