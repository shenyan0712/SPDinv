#define dtype double

/*
计算下三角块矩阵的逆, 存到in的上三角
本kernel只计算第i条斜线块，然后调用计算下一个斜线块i+1的kernel_trigMatInv
*/
__kernel void kern_trigMat_inv_m1(
	__global dtype *in,
	__global dtype *diagBlk,	//来自cholesky计算的Lii^-1
	__global dtype *ret,
	const int mat_size,		//原始矩阵的尺寸
	const int ii			//第ii条斜线块
)
{
	int addr, addr2, ijuv_addr, jivu_addr;

	int tr = get_global_id(0);
	int j = (int)(tr / 3);		//当前处理块的列号
	int i = ii + j;
	int u = tr - j * 3;			//当前处理块的第u行
	int v = get_global_id(1);	//当前处理块的第v列

	dtype tt;
	dtype L0, L1, L2, L3, L4, L5;

	ijuv_addr = (i * 3 + u)*mat_size + j * 3 + v;
	jivu_addr = (j * 3 + v)*mat_size + i * 3 + u;

	//如果是对角线块，从diagBlk取其逆Lii^-1，存到out的对角线上
	if (i == j)
	{
		in[ijuv_addr] = diagBlk[i*9+ v* 3 + u];
		barrier(CLK_GLOBAL_MEM_FENCE);	//同步，以使从mat读取完毕
	}
	else
	{
		//对于非对角块
		//1，计算T_ij=sum(L_ik*X_kj), k=j to i-1,  注意X_kj^t是在out的上三角
		tt = 0.0;
		for (int k = j; k < i; k++)
		{
			//L_ik的u行与 X_kj^t的v行的点积
			addr = (i * 3 + u)*mat_size + k * 3;
			addr2 = (j * 3 + v)*mat_size + k * 3;
			tt += in[addr++] * in[addr2++];
			tt += in[addr++] * in[addr2++];
			tt += in[addr] * in[addr2];
		}
		in[jivu_addr] = tt;		//T_ij^t的u,v元素存到in的上三角

		//同步
		barrier(CLK_GLOBAL_MEM_FENCE);

		//计算X_ij=(L_ii^-1)*T_ij, 而存储的是L_ii^-t 和T_ij^t, 那么应该取L_ii^-t的u列与T_ij^t的v行的点积
		tt = 0.0;
		addr = i * 3 * mat_size + i * 3 + u;		//u列第一个元素
		addr2 = (j * 3 + v)*mat_size + i * 3;	//v行第一个元素
		tt += in[addr] * in[addr2++];
		tt += in[addr + mat_size] * in[addr2++];
		tt += in[addr + 2 * mat_size] * in[addr2];

		//同步
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (i != j) {
			in[jivu_addr] = -tt;
			//in[ijuv_addr] = 0;
		}
	}

	//调用ii+1条斜线块的计算的kernel
	addr = mat_size - (ii + 1) * 3;		//剩余块的总行数
	if (tr == 0 && v == 0 && addr>0)
	{
		void(^kernel_trigMatInv_wrapper)(void) =
			^{
			kern_trigMat_inv_m1(in,diagBlk, ret, mat_size, ii + 1);
		};
		size_t    global_size[2] = { addr, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kernel_trigMatInv_wrapper
		);
	}
}
