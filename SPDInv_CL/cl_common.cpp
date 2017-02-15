#include"stdafx.h"


#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>
#include <CL/cl.h>

#include "cl_spd_inv.h"
#include "cl_common.h"

using namespace std;


/* auxiliary memory allocation routine with error checking */
void *emalloc_(char *file, int line, size_t sz)
{
	void *ptr;

	ptr = (void *)malloc(sz);
	if (ptr == NULL) {
		fprintf(stderr, "SBA: memory allocation request for %zu bytes failed in file %s, line %d, exiting", sz, file, line);
		exit(1);
	}

	return ptr;
}

void printBuf2D(FILE *file, cl_command_queue queue, cl_mem buf, int rsize, int csize, char *title)
{
	dtype *ptr;
	int bufSize = sizeof(dtype)*rsize*csize;
	ptr = (dtype*)emalloc(bufSize);

	clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufSize, ptr, 0, NULL, NULL);

	fprintf(file, "%s:\n", title);
	for (int i = 0; i < rsize; i++)
	{
		for (int j = 0; j < csize; j++)
		{
			fprintf(file, "%.12lf,  ", ptr[i*csize + j]);
		}
		fprintf(file, "\n");
	}
	free(ptr);
}

void printBuf1D(FILE *file, cl_command_queue queue, cl_mem buf, int size, char *title)
{
	dtype *ptr;
	int bufSize = sizeof(dtype)*size;
	ptr = (dtype*)emalloc(bufSize);

	clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufSize, ptr, 0, NULL, NULL);

	fprintf(file, "%s:\n", title);
	for (int i = 0; i < size; i++)
	{
		fprintf(file, "%lf,  ", ptr[i]);
	}
	fprintf(file, "\n");

	free(ptr);
}


/** 读取文件并将其转为字符串 */
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return -1;
}


void checkErr(cl_int err, const char* file, int num)
{
	if (CL_SUCCESS != err)
	{
		printf("OpenCL error(%d) at file %s(%d).\n", err, file, num - 1);
		system("Pause");
		exit(EXIT_FAILURE);
	}
}

/* 找到第一个平台，返回第一个设备ID  */
cl_device_id get_first_gpu() {

	cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_device_id dev;
	int err;

	//获取平台的个数
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(err, __FILE__, __LINE__);

	//根据个数创建cl_platform_id对象
	platformIDs = (cl_platform_id *)malloc(
		sizeof(cl_platform_id) * numPlatforms);

	//获取平台ID
	err = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(err, __FILE__, __LINE__);

	/* 获取第一个设备的ID */
	err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platformIDs[1], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	}
	checkErr(err, __FILE__, __LINE__);
	return dev;
}

/* 找到第一个平台，返回第一个设备ID  */
cl_device_id get_first_cpu() {

	cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_device_id dev;
	int err;

	//获取平台的个数
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(err, __FILE__, __LINE__);

	//根据个数创建cl_platform_id对象
	platformIDs = (cl_platform_id *)malloc(
		sizeof(cl_platform_id) * numPlatforms);

	//获取平台ID
	err = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(err, __FILE__, __LINE__);

	/* 获取第一个设备的ID */
	err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platformIDs[1], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	checkErr(err, __FILE__, __LINE__);
	return dev;
}



/* 创建并编建程序对象 */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename)
{

	cl_program program;
	char *program_log;
	size_t program_size, log_size;
	int err;

	/* 读取程序文件 */
	string sourceStr;
	err = convertToString(filename, sourceStr);
	const char *program_str = sourceStr.c_str();
	program_size = strlen(program_str);

	/* 从程序文件创建程序对象 */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_str, &program_size, &err);
	checkErr(err, __FILE__, __LINE__);

	/* 编程程序 */
	err = clBuildProgram(program, 1, &dev, "-cl-std=CL2.0 -D CL_VERSION_2_0", NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		system("pause");
		exit(1);
	}
	return program;
}

