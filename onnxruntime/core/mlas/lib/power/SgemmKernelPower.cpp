/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/
#include "SgemmKernelpower.h"

void MlasSgemmTransposePackB16x4VSX(
    float* D,
    const float* B,
    size_t ldb
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    4 columns of 16 rows from the source matrix are transposed to 16 columns of 4
    rows in the destination packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

Return Value:

    None.

--*/
{
  long int i, j;
  typedef __vector float VA;
  float *aoffset = NULL, *boffset = NULL;
  float  *aoffset1 = NULL, *aoffset2 = NULL, *aoffset3 = NULL, *aoffset4 = NULL;
  float *aoffset5 = NULL, *aoffset6 = NULL, *aoffset7 = NULL, *aoffset8 = NULL;
  float  *aoffset9 = NULL, *aoffset10 = NULL, *aoffset11 = NULL, *aoffset12 = NULL;
  float *aoffset13 = NULL, *aoffset14 = NULL, *aoffset15 = NULL, *aoffset16 = NULL;
  VA c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
  VA c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
  VA c9[2] = {0}, c10[2] = {0}, c11[2] = {0}, c12[2] = {0};
  VA c13[2] = {0}, c14[2] = {0}, c15[2] = {0}, c16[2] = {0};
  VA t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16;
  aoffset = const_cast<float *>(B);
  boffset = D;
      aoffset1 = aoffset;
      aoffset2 = aoffset1 + ldb;
      aoffset3 = aoffset2 + ldb;
      aoffset4 = aoffset3 + ldb;
      aoffset5 = aoffset4 + ldb;
      aoffset6 = aoffset5 + ldb;
      aoffset7 = aoffset6 + ldb;
      aoffset8 = aoffset7 + ldb;
      aoffset9 = aoffset8 + ldb;
      aoffset10 = aoffset9 + ldb;
      aoffset11 = aoffset10 + ldb;
      aoffset12 = aoffset11 + ldb;
      aoffset13 = aoffset12 + ldb;
      aoffset14 = aoffset13 + ldb;
      aoffset15 = aoffset14 + ldb;
      aoffset16 = aoffset15 + ldb;
        c1[0] = vec_xl(0, aoffset1);
        c2[0] = vec_xl(0, aoffset2);
        c3[0] = vec_xl(0, aoffset3);
        c4[0] = vec_xl(0, aoffset4);
        c5[0] = vec_xl(0, aoffset5);
        c6[0] = vec_xl(0, aoffset6);
        c7[0] = vec_xl(0, aoffset7);
        c8[0] = vec_xl(0, aoffset8);
	c9[0] = vec_xl(0, aoffset9);
	c10[0] = vec_xl(0, aoffset10);
	c11[0] = vec_xl(0, aoffset11);
        c12[0] = vec_xl(0, aoffset12);
	c13[0] = vec_xl(0, aoffset13);
	c14[0] = vec_xl(0, aoffset14);
	c15[0] = vec_xl(0, aoffset15);
        c16[0] = vec_xl(0, aoffset16);

        t1 = vec_mergeh(c1[0], c2[0]);
        t2 = vec_mergeh(c3[0], c4[0]);
        t3 = vec_mergeh(c5[0], c6[0]);
        t4 = vec_mergeh(c7[0], c8[0]);
	t5 = vec_mergeh(c9[0], c10[0]);
	t6 = vec_mergeh(c11[0], c12[0]);
	t7 = vec_mergeh(c13[0], c14[0]);
	t8 = vec_mergeh(c15[0], c16[0]);

        t9 = vec_xxpermdi(t1, t2, 0);
        t10 = vec_xxpermdi(t3, t4, 0);
        t11 = vec_xxpermdi(t5, t6, 0);
	t12 = vec_xxpermdi(t7, t8, 0);

        t13 = vec_xxpermdi(t1, t2, 3);
        t14 = vec_xxpermdi(t3, t4, 3);
	t15 = vec_xxpermdi(t5, t6, 3);
	t16 = vec_xxpermdi(t7, t8, 3);
        vec_xst(t9, 0, boffset);
        vec_xst(t10, 0, boffset + 4);
        vec_xst(t11, 0, boffset + 8);
        vec_xst(t12, 0, boffset + 12);
	vec_xst(t13, 0, boffset + 16);
	vec_xst(t14, 0, boffset + 20);
	vec_xst(t15, 0, boffset + 24);
	vec_xst(t16, 0, boffset + 28);

        t1 = vec_mergel(c1[0], c2[0]);
        t2 = vec_mergel(c3[0], c4[0]);
        t3 = vec_mergel(c5[0], c6[0]);
        t4 = vec_mergel(c7[0], c8[0]);
	t5 = vec_mergel(c9[0], c10[0]);
	t6 = vec_mergel(c11[0], c12[0]);
	t7 = vec_mergel(c13[0], c14[0]);
	t8 = vec_mergel(c15[0], c16[0]);

        t9 = vec_xxpermdi(t1, t2, 0);
        t10 = vec_xxpermdi(t3, t4, 0);
	t11 = vec_xxpermdi(t5, t6, 0);
	t12 = vec_xxpermdi(t7, t8, 0);
        t13 = vec_xxpermdi(t1, t2, 3);
        t14 = vec_xxpermdi(t3, t4, 3);
	t15 = vec_xxpermdi(t5, t6, 3);
	t16 = vec_xxpermdi(t7, t8, 3);

        vec_xst(t9, 0, boffset + 32);
        vec_xst(t10, 0, boffset + 36);
        vec_xst(t11, 0, boffset + 40);
        vec_xst(t12, 0, boffset + 44);
	vec_xst(t13, 0, boffset + 48);
	vec_xst(t14, 0, boffset + 52);
	vec_xst(t15, 0, boffset + 56);
	vec_xst(t16, 0, boffset + 60);
}

size_t
MLASCALL
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;

    MLAS_FLOAT32X4 AlphaBroadcast = MlasBroadcastFloat32x4(alpha);

    if (CountM >= 4) {
        RowsHandled = MlasSgemmProcessCount<4>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasSgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }

    return RowsHandled;
}
