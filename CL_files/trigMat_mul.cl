#define dtype double


__kernel void kern_fill_rest(
	__global dtype *in,
	__global dtype *diag,
	const int mat_size);




/*
计算S^-1=(L^-t)*(L^-1)，
in为上三角块矩阵L^-t
*/
__kernel void kern_trigMat_mul(
	__global dtype *in,
	__global dtype *diag,
	const int mat_size)
{
	dtype val;
	int u = get_global_id(0);		//本工作项处理的元素的行号
	int v = get_global_id(1);		//本工作项处理的元素的列号

	if (v > u) return;

	//(i,j)的值为in矩阵的i行与j行的乘积。
	int u_addr, v_addr;
	u_addr = u*mat_size + u;		//从i列开始，之前的相乘都为0
	v_addr = v*mat_size + u;
	val = 0;
	for (int k = u; k < mat_size; k++)
	{
		val += in[u_addr++] * in[v_addr++];
	}
	if (u == v)
		diag[u] = val;
	else
		in[u*mat_size + v] = val;

	//call kernel to fill diagonal element
	if (u == 0 && v == 0)
	{
		void(^kernel_fill_rest_wrapper)(void) =
			^{
			kern_fill_rest(in,diag, mat_size);
		};
		size_t    global_size[2] = { mat_size, mat_size};
		//size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kernel_fill_rest_wrapper
		);

	}
}



/*
1,写入对角元素
2,将下三角元素复制到上三角
*/
__kernel void kern_fill_rest(
	__global dtype *in,
	__global dtype *diag,
	const int mat_size)
{
	int addr1, addr2;
	int u = get_global_id(0);		//本工作项处理的元素的行号
	int v = get_global_id(1);		//本工作项处理的元素的列号

	if (v > u) return;
	
	addr1 = u*mat_size + v;
	if (u == v)
	{
		in[addr1] = diag[u];
	}
	else {
		addr2 = v*mat_size + u;
		in[addr2] = in[addr1];
	}

}