#pragma once

#include <CL/cl.h>
#include <string>

using namespace std;

#define emalloc(sz)       emalloc_(__FILE__, __LINE__, sz)

void checkErr(cl_int err, const char* file, int num);

cl_device_id get_first_gpu();
cl_device_id get_first_cpu();
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

void printBuf2D(FILE *file, cl_command_queue queue, cl_mem buf, int rsize, int csize, char *title);
void printBuf1D(FILE *file, cl_command_queue queue, cl_mem buf, int size, char *title);