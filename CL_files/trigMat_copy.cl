
/*
将上三角拷贝到下三角, 上三角置零
*/
__kernel void kern_trigMat_copy(
	__global dtype *in,
	__global dtype *out,
	const int mat_size)
{
	dtype val;
	int u = get_global_id(0);		//本工作项处理的元素的行号
	int v = get_global_id(1);		//本工作项处理的元素的列号


	if (v > u) {
		out[u*mat_size + v] = 0;
	}
	else
	{
		out[u*mat_size + v] = in[v*mat_size + u];
	}
}