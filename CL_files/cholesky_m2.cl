__kernel void kern_cholesky_m6(
	__global dtype *mat,
	__global dtype *diagInv,	//
	__global dtype *ret,
	const int mat_size,
	__local dtype *T,
	__local dtype *L,
	__local int *si)
{
	dtype sum, t1, t2;
	int ijuv_addr, jivu_addr, addr1, addr2, addr3;
	dtype3 *ptr1, *ptr2;
	int gidx = get_global_id(0);	//
	int gidy = get_global_id(1);	//
	int gi = gidx / 3;
	int u = gidx - gi * 3;
	int v = gidy;
	int i;
	int numOfBlkCols = mat_size / 3;
	int numOfWorkBlks = get_global_size(0) / 3;		//一次能处理的块数
	//int diag_addr = u * 3 + v;
	for (int j = 0; j < numOfBlkCols; j++)
	{
		i = gi + j;
		/****************处理该列的对角块***************/
		//1, 计算对角块Ljj的uv元素
		if (i == j)
		{
			ijuv_addr = (j * 3 + u)*mat_size + j * 3 + v;

			//1.1 *****计算Tij_uv
			sum = mat[ijuv_addr];	//Aij_uv元素
			for (int k = 0; k < j; k++)
			{
				addr1 = (j * 3 + u)*mat_size + k * 3;		//Lik的第u行
				addr2 = (j * 3 + v)*mat_size + k * 3;		//Ljk的第v行
				ptr1 = (dtype3 *)&mat[addr1];
				ptr2 = (dtype3 *)&mat[addr2];
				sum -= dot(*ptr1, *ptr2);
			}
			addr1 = u * 3 + v;
			T[addr1] = sum;		//保存到本地缓存
			
			//1.2 *****计算Ljj的uv元素
			switch (addr1)
			{
			case 0:		//L00 or L(0)
				L[0] = sqrt(T[0]);
				mat[ijuv_addr] = L[0];
				if (!isfinite(L[0])) *ret = 1.0;
				break;
			case 1:		//L01,L02
			case 2:
				mat[ijuv_addr] = 0;
				break;
			case 3:		//L10
				L[1] = T[1 * 3 + 0] / sqrt(T[0]);
				mat[ijuv_addr] = L[1];
				if (!isfinite(L[1])) *ret = 1.0;
				break;
			case 4:		//L11
				L[2] = sqrt(T[1 * 3 + 1] - T[1 * 3 + 0] * T[1 * 3 + 0] / T[0]);;
				mat[ijuv_addr] = L[2];
				if (!isfinite(L[2])) *ret = 1.0;
				break;
			case 5:		//L12
				mat[ijuv_addr] = 0;
				break;
			case 6:		//L20 or L(3)
				L[3] = T[2 * 3 + 0] / sqrt(T[0]);
				mat[ijuv_addr] = L[3];
				if (!isfinite(L[3])) *ret = 1.0;
				break;
			case 7:		//L21 or L(4)
				L[4] = sqrt(T[0] / (T[0] * T[1 * 3 + 1] - T[1 * 3 + 0] * T[1 * 3 + 0]))*(T[2 * 3 + 1] - T[1 * 3 + 0] * T[2 * 3 + 0] / T[0]);
				mat[ijuv_addr] = L[4];
				if (!isfinite(L[4])) *ret = 1.0;
				break;
			case 8:		//L22 or (5)
				t1 = T[0] / (T[0] * T[1 * 3 + 1] - T[1 * 3 + 0] * T[1 * 3 + 0]);
				t2 = T[2 * 3 + 1] - T[1 * 3 + 0] * T[2 * 3 + 0] / T[0];
				t2 = t2*t2;
				L[5] = sqrt(T[2 * 3 + 2] - T[2 * 3 + 0] * T[2 * 3 + 0] / T[0] - t1*t2);
				mat[ijuv_addr] = L[5];
				if (!isfinite(L[5])) *ret = 1.0;
				break;
			default:
				break;
			}	//end of switch
		}
		barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

		//计算对角块的逆矩阵Ljj^-1的uv元素
		if (i == j)
		{
			addr1 = u * 3 + v;
			switch (addr1)
			{
			case 0:		//L00 or L(0)
				t1 = 1 / L[0];
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			case 1:		//L01,L02
			case 2:
				diagInv[addr1] = 0;
				break;
			case 3:		//L10 or L(1)
				t1 = -L[1] / (L[0] * L[2]);
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			case 4:		//L11 or L(2)
				t1 = 1 / L[2];
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			case 5:		//L12
				diagInv[addr1] = 0;
				break;
			case 6:		//L20 or L(3)
				t1 = (L[1] * L[4] - L[2] * L[3]) / (L[0] * L[2] * L[5]);
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			case 7:		//L21 or L(4)
				t1 = -L[4] / (L[2] * L[5]);
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			case 8:		//L22 or L(5)
				t1 = 1 / L[5];
				diagInv[addr1] = t1;
				if (!isfinite(t1)) *ret = 1.0;
				break;
			default:
				break;
			}	//end of switch() for matrix inverse
		}

		if (gidx == 0 && gidy == 0)
			*si = j + 1;
		barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
											/****************处理剩余块******************/
		while (*si<numOfBlkCols)	//剩余块中，一次只能处理numOfWorkBlks个块
		{
			i = *si + gi;
			ijuv_addr = (i * 3 + u)*mat_size + j * 3 + v;
			jivu_addr = (j * 3 + v)*mat_size + i * 3 + u;

			//1,计算Tij = Aij - sum(Lik*Ljk^t)中的uv元素
			if (i<numOfBlkCols)		//确保块的行号不会超出
			{
				sum = mat[ijuv_addr];	//Aij_uv元素
				for (int k = 0; k < j; k++)
				{
					addr1 = (i * 3 + u)*mat_size + k * 3;		//Lik的第u行
					addr2 = (j * 3 + v)*mat_size + k * 3;		//Ljk的第v行
																//sum -= mat[addr1++] * mat[addr2++];
																//sum -= mat[addr1++] * mat[addr2++];
																//sum -= mat[addr1] * mat[addr2];
					ptr1 = (dtype3*)&mat[addr1];
					ptr2 = (dtype3*)&mat[addr2];
					sum -= dot(*ptr1, *ptr2);
				}
				mat[ijuv_addr] = sum;		//保存Tij_uv到原矩阵
			}
			barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

												//2,计算Lij=Tij*(Ljj^-t)中的uv元素
			if (i < numOfBlkCols)
			{
				addr1 = (i * 3 + u)*mat_size + j * 3;	//Tij的第u行
				addr2 = v * 3;				//(Ljj^-1)的第v行
				sum = 0.0;
				//sum += mat[addr1++] * diagInv[addr2++];
				//sum += mat[addr1++] * diagInv[addr2++];
				//sum += mat[addr1] * diagInv[addr2];
				ptr1 = (dtype3*)&mat[addr1];
				ptr2 = (dtype3*)&diagInv[addr2];
				sum += dot(*ptr1, *ptr2);
			}
			barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
												//3, 将计算出的值存回原矩阵
			if (i < numOfBlkCols)
			{
				mat[ijuv_addr] = sum;	// sum;
				mat[jivu_addr] = 0;
			}

			if (gidx == 0 && gidy == 0)
				*si = *si + numOfWorkBlks;
			barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
		}
	}

}	//end of kernel