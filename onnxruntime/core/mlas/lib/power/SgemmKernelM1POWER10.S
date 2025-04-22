/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelM1POWER10.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM). This handles the special case of M=1.

    This implementation uses AVX instructions.

--*/
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows. This handles the special case of M=1.

    The elements in matrix B are not transposed.

Arguments:

    A (r3) - Supplies the address of matrix A.

    B (r4) - Supplies the address of matrix B.

    C (r5) - Supplies the address of matrix C.

    CountK (r6) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountN (r7) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    Beta (r8) - Supplies the first dimension of matrix B.

Return Value:

    None.

--*/
#include "asmmacro.h"
.text
FUNCTION_ENTRY MlasSgemmKernelM1POWER10
	cmpldi 7,0    /* r7 is N.  */
	beq L_exit
L_loop_N:
	mr 11,3       /* copy A matrix(r3) address to r11.  */
	mr 12,6       /* copy K 9(r6) to r12.  */
/* Reset acc0-3.  */
	xxsetaccz 0
	xxsetaccz 1
	xxsetaccz 2
	xxsetaccz 3
	cmpldi 12,4	/* r6 is K.  */
        blt L_gerpp
L_multi_gerpp:
	lxv 32,0(11)	/* vs32 ,4 floats of A and vs33-vs35 for vec_splats.  */
	xxspltw 33,32,2
	xxspltw 34,32,1
	xxspltw 35,32,0
/* Load 4 vector elements of B matrix vs36 to vs39.  */
	lxvp 36,0(4)
	lxvp 38,32(4)
/* Multiply-Add-Accumulate.  */
	xvf32gerpp 0,32,37
	xvf32gerpp 1,32,36
	xvf32gerpp 2,32,39
	xvf32gerpp 3,32,38
/* Load next 4 vector elements of B.  */
        lxvp 36,64(4)
	lxvp 38,96(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,33,37
        xvf32gerpp 1,33,36
        xvf32gerpp 2,33,39
        xvf32gerpp 3,33,38
/* Load next 4 vector elements of B.  */
        lxvp 36,128(4)
        lxvp 38,160(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,34,37
        xvf32gerpp 1,34,36
        xvf32gerpp 2,34,39
        xvf32gerpp 3,34,38
/* Load next 4 vector elements of B.  */
        lxvp 36,192(4)
        lxvp 38,224(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,35,37
        xvf32gerpp 1,35,36
        xvf32gerpp 2,35,39
        xvf32gerpp 3,35,38
/* Update loop count A(r3),K(r6) and B(r4).  */
	addi 11,11,16
	addi 4,4,256
	addi 12,12,-4
	cmpldi 12,4
	bge L_multi_gerpp
L_gerpp:
	cmpldi 12,1
        blt L_unprime_acc

	lfs 31,0(11)
	xscvdpspn  33,31
	xxspltw 33,33,0
/* Load 4 vector elements of B matrix.  */
        lxvp 36,0(4)
        lxvp 38,32(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,33,37
        xvf32gerpp 1,33,36
        xvf32gerpp 2,33,39
        xvf32gerpp 3,33,38
/* Update loop count A ,K and B.  */
	addi 4,4,64
	addic. 12,12,-1
	beq L_unprime_acc
        lfs 31,4(11)
        xscvdpspn  33,31
        xxspltw 33,33,0
/* Load 4 vector elements of B matrix.  */
        lxvp 36,0(4)
        lxvp 38,32(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,33,37
        xvf32gerpp 1,33,36
        xvf32gerpp 2,33,39
        xvf32gerpp 3,33,38      
	addi 4,4,64
        addic. 12,12,-1
        beq L_unprime_acc
        lfs 31,8(11)
        xscvdpspn  33,31
        xxspltw 33,33,0
/* Load 4 vector elements of B matrix.  */
        lxvp 36,0(4)
        lxvp 38,32(4)
/* Multiply-Add-Accumulate.  */
        xvf32gerpp 0,33,37
        xvf32gerpp 1,33,36
        xvf32gerpp 2,33,39
        xvf32gerpp 3,33,38
	addi 4,4,64
L_unprime_acc:
	cmpldi 8,1
        bne L_beta_not_zero
	cmpldi 7,16
	blt L_unprime_partial
	xxmfacc 0
	xxmfacc 1
	xxmfacc 2
	xxmfacc 3
/* storing vs3,vs7,vs11 and vs15 to the C.  */
	stxv 3,0(5)
	stxv 7,16(5)
	stxv 11,32(5)
	stxv 15,48(5)
	b L_update_C
L_unprime_partial:
	cmpldi 7,12
	blt L_unprime_partial_N8
/* when N is 12 then store the vs3,vs7 and vs11 to C.  */
	xxmfacc 0
        xxmfacc 1
        xxmfacc 2
        stxv 3,0(5)
        stxv 7,16(5)
        stxv 11,32(5)
        addi 5,5,48
	andi. 7,7,3
	beq L_exit
        xxmfacc 3
        xxsldwi 15,15,15,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 15,15,15,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 15,15,15,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,8(5)
        b L_exit	 

L_unprime_partial_N8:
	cmpldi 7,8
	blt L_unprime_partial_N4
	xxmfacc 0
	xxmfacc 1
	stxv 3,0(5)
        stxv 7,16(5)
	addi 5,5,32
	andi. 7,7,3
	beq L_exit
	xxmfacc 2
        xxsldwi 11,11,11,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 11,11,11,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 11,11,11,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,8(5)
        b L_exit

L_unprime_partial_N4:
        cmpldi 7,4
        blt L_unalign_N
        xxmfacc 0
        stxv 3,0(5)
        addi 5,5,16
        andi. 7,7,3
        beq L_exit
        xxmfacc 1
        xxsldwi 7,7,7,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,7   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 7,7,7,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,7   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 7,7,7,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,7   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,8(5)
        b L_exit

L_unalign_N:
	xxmfacc 0
        xxsldwi 3,3,3,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,0(5)
	addic. 7,7,-1
	beq L_exit
        xxsldwi 3,3,3,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,4(5)
        addic. 7,7,-1
	beq L_exit
        xxsldwi 3,3,3,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        stfs 0,8(5)
	b L_exit	
L_beta_not_zero:
        cmpldi 7,16
        blt L_beta_unprime_partial
        xxmfacc 0
        xxmfacc 1
        xxmfacc 2
        xxmfacc 3
        lxvp 32,0(5)
	lxvp 34,32(5)
        xvaddsp 3,3,33
	xvaddsp 7,7,32
	xvaddsp 11,11,35
	xvaddsp 15,15,34
/* storing vs3,vs7,vs11 and vs15 to the C.  */
        stxv 3,0(5)
        stxv 7,16(5)
        stxv 11,32(5)
        stxv 15,48(5)
        b L_update_C
L_beta_unprime_partial:
        cmpldi 7,12
        blt L_beta_unprime_partial_N8
/* when N is 12 then store the vs3,vs7 and vs11 to C.  */
        xxmfacc 0
        xxmfacc 1
        xxmfacc 2
	lxvp 32,0(5)
	lxv 34,32(5)
        xvaddsp 3,3,33
        xvaddsp 7,7,32
        xvaddsp 11,11,34
        stxv 3,0(5)
        stxv 7,16(5)
        stxv 11,32(5)
        addi 5,5,48
        andi. 7,7,3
        beq L_exit
        xxmfacc 3
        xxsldwi 15,15,15,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,0(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 15,15,15,3
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,4(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 15,15,15,3
        xscvspdpn 0,15   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,8(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,8(5)
        b L_exit

L_beta_unprime_partial_N8:
        cmpldi 7,8
        blt L_beta_unprime_partial_N4
/* when N is 8 then store the vs3,vs7 to C.  */
        xxmfacc 0
        xxmfacc 1
        lxvp 32,0(5)
        xvaddsp 3,3,33
        xvaddsp 7,7,32
        stxv 3,0(5)
        stxv 7,16(5)
        addi 5,5,32
        andi. 7,7,3
        beq L_exit
        xxmfacc 2
        xxsldwi 11,11,11,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,0(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 11,11,11,3
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,4(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 11,11,11,3
        xscvspdpn 0,11   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,8(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,8(5)
        b L_exit
L_beta_unprime_partial_N4:
	cmpldi 7,4
	blt L_beta_unalign_N
	xxmfacc 0
	lxv 32,0(5)
	xvaddsp 3,3,32
	stxv 3,0(5)
	addi 5,5,16
	andi. 7,7,3
	beq L_exit
	xxmfacc 1
	xxsldwi 7,7,7,3	/*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
	xscvspdpn 0,7	/* moving lower 32 to vso so the fp0 is also will be updated.  */
	lfs 10,0(5)	/* loading float from $r5 address.  */
	fadds 10,10,0
	stfs 10,0(5)
        addic. 7,7,-1
        beq L_exit
	xxsldwi 7,7,7,3
        xscvspdpn 0,7   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,4(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 7,7,7,3
	xscvspdpn 0,7   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,8(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,8(5) 
	b L_exit
L_beta_unalign_N:
        xxmfacc 0
        xxsldwi 3,3,3,3 /*leftshift the float places by 3 times (1 2 3 4) will be (2 3 4 1).  */
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,0(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,0(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 3,3,3,3
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,4(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,4(5)
        addic. 7,7,-1
        beq L_exit
        xxsldwi 3,3,3,3
        xscvspdpn 0,3   /* moving lower 32 to vso so the fp0 is also will be updated.  */
        lfs 10,8(5)     /* loading float from $r5 address.  */
        fadds 10,10,0
        stfs 10,8(5) 
        b L_exit
L_update_C:
	addi 5,5,64
        addic. 7,7,-16
	bgt L_loop_N
L_exit:
	blr
.end
