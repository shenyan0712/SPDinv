
#include"stdafx.h"
#include <iostream>
#include <fstream>

#include "cl_common.h"
#include "cl_spd_inv.h"

using namespace std;

/*
配置好openCL库, 填写cl_device结构体
*/
void cl_SPDInv_setup(SPDInv_structPtr SPDInvPtr, int maxsize, int maxblksize)
{
	cl_int err;
	SPDInvPtr->device = get_first_cpu();
	//SPDInvPtr->device = get_first_cpu();
	// 为设备创建上下文
	SPDInvPtr->context = clCreateContext(NULL, 1, &SPDInvPtr->device, NULL, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	// 创建命令对队
	SPDInvPtr->queue = clCreateCommandQueueWithProperties(SPDInvPtr->context, SPDInvPtr->device, 0, &err);
	checkErr(err, __FILE__, __LINE__);

	/* 在设备上创建一个命令队列 */
	cl_queue_properties properties_gpu[] = {
		CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
		CL_QUEUE_SIZE,8192,
		0 };
	SPDInvPtr->queue_dev = clCreateCommandQueueWithProperties(SPDInvPtr->context, SPDInvPtr->device, properties_gpu, &err);
	checkErr(err, __FILE__, __LINE__);


	//创建相关缓存
	//##########创建输入buffer###########//
	SPDInvPtr->buf_spd_A = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(maxsize*maxsize), NULL, &err);
	SPDInvPtr->buf_spd_B = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(maxsize*maxsize), NULL, &err);
	SPDInvPtr->buf_diagAux = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(maxblksize*maxsize), NULL, &err);
	SPDInvPtr->buf_blkBackup = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(maxblksize*maxsize), NULL, &err);
	SPDInvPtr->buf_diag = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(maxsize), NULL, &err);
	SPDInvPtr->buf_ret = clCreateBuffer(SPDInvPtr->context, CL_MEM_READ_WRITE, sizeof(dtype), NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	/********************************************************************************/
	/********************************************************************************/
	//
	// 创建程序对象
	SPDInvPtr->program = build_program(SPDInvPtr->context, SPDInvPtr->device, KERN_FILE);

	SPDInvPtr->kern_cholesky_m1 = clCreateKernel(SPDInvPtr->program, "kern_cholesky_m1", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_trigMat_inv_m1 = clCreateKernel(SPDInvPtr->program, "kern_trigMat_inv_m1", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_trigMat_mul = clCreateKernel(SPDInvPtr->program, "kern_trigMat_mul", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_trigMat_copy = clCreateKernel(SPDInvPtr->program, "kern_trigMat_copy", &err);
	checkErr(err, __FILE__, __LINE__);

	SPDInvPtr->kern_cholesky_mod = clCreateKernel(SPDInvPtr->program, "kern_cholesky_mod", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_cholmod_E = clCreateKernel(SPDInvPtr->program, "kern_cholmod_E", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_cholmod_blk = clCreateKernel(SPDInvPtr->program, "kern_cholmod_blk", &err);
	checkErr(err, __FILE__, __LINE__);
	SPDInvPtr->kern_mat_max = clCreateKernel(SPDInvPtr->program, "kern_mat_max", &err);
	checkErr(err, __FILE__, __LINE__);

	SPDInvPtr->kern_gen_rand = clCreateKernel(SPDInvPtr->program, "kern_gen_rand", &err);
	checkErr(err, __FILE__, __LINE__);
}


void cl_SPDInv_release(SPDInv_structPtr SPDInvPtr)
{



}