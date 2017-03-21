#pragma once

#include <CL/cl.h>

#include "spd_inv.h"

#define KERN_FILE "../CL_files/spd_inv.cl"

#define DEV_TYPE	CL_DEVICE_TYPE_CPU

#define DEBUG_GEN_MAT		0
#define DEBUG_CHOLESKY		0
#define DEBUG_TRIGMAT_INV	0
#define DEBUG_TRIGMAT_MUL	0


typedef struct {
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_command_queue queue_dev;		//queue on device
	cl_program program;
	cl_program program_cpu;

	//buffer associated to LM
	cl_mem buf_spd_A;		//存放原始矩阵， 及计算结果逆矩阵
	cl_mem buf_spd_B;		//
	cl_mem buf_diagAux;			//辅助buffer, 尺寸>blksize*matsize
	cl_mem buf_blkBackup;
	cl_mem buf_diag;		//cholmod使用
	cl_mem buf_ret;

	cl_kernel kern_cholesky_m1;
	cl_kernel kern_trigMat_inv_m1;
	cl_kernel kern_trigMat_mul;
	cl_kernel kern_trigMat_copy;

	cl_kernel kern_mat_max;
	cl_kernel kern_cholesky_mod;		//带修正的cholesky
	cl_kernel kern_cholmod_E;
	cl_kernel kern_cholmod_blk;			//带修正的cholesky, 分块形式
	
	cl_kernel kern_gen_rand;


}SPDInv_struct, *SPDInv_structPtr;

//maxsize为方阵的最大尺寸, maxblksize为子块的最大尺寸
void cl_SPDInv_setup(SPDInv_structPtr SPDInvPtr, int maxsize,int maxblksize);
void cl_SPDInv_release(SPDInv_structPtr SPDInvPtr);

dtype cholesky_m1(cl_command_queue queue, cl_kernel kern_cholesky,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf,
	int matSize, dtype *outMat);

void get_delta_beta(cl_command_queue queue, cl_kernel kern_delta_beta, cl_mem matBuf, cl_mem diagAuxBuf,
	int matsize, dtype *delta, dtype *beta);
void cholesky_mod(cl_command_queue queue, cl_kernel kern_cholesky_mod, cl_kernel kern_delta_beta,
	cl_mem matBuf, cl_mem auxBuf,cl_mem diagBuf,cl_mem retBuf,
	int matSize, dtype *outMat, bool copyOut);

void compute_cholmod_E(cl_command_queue queue, cl_kernel kern_cholmod_E,
	cl_mem matBuf, cl_mem E_Buf, int matSize, dtype *Eout);

dtype trigMat_inv_m1(cl_command_queue queue, cl_kernel kern_trigMat_inv, cl_mem matBuf, cl_mem auxBuf,
	cl_mem retBuf, int mat_size, dtype *outMat);
void trigMat_mul(cl_command_queue queue, cl_kernel kern_trigMat_mul, cl_mem matBuf, cl_mem auxBuf, cl_mem outBuf,
	int mat_size, dtype *outMat);
void trigMat_copy(cl_command_queue queue, cl_kernel kern_trigMat_copy, cl_mem inBuf, cl_mem outBuf,
	int mat_size, dtype *outMat);

void cholmod_blk(cl_command_queue queue, cl_kernel kern_cholmod_blk, cl_kernel kern_mat_max,
	cl_mem matBuf, cl_mem blkBackupBuf, cl_mem diagBlkAuxBuf, cl_mem diagBuf, cl_mem retBuf,
	int matSize, dtype *outMat);

void gen_rand(cl_command_queue queue, cl_kernel kern_gen_rand, cl_mem outBuf,
	int mat_size, dtype *outMat);