/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_power12.cpp

Abstract:

    This module implements QGEMM kernel for POWER12 with MMA+ support.
    Uses 16x16 block size with __dmr1024 accumulators.

--*/

#include <inttypes.h>

#include "mlasi.h"
#include "qgemm.h"


// POWER10 kernel definition for M < 8 fallback
struct MLAS_GEMM_QUANT_KERNEL_POWER10
{
    typedef int8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef uint8_t OffsetBType;
    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 16, 256, 384 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 16, 128, 128 };
};

// Forward declarations of POWER10 functions
template <>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
);

template <>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
);

template <>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
);


struct MLAS_GEMM_QUANT_KERNEL_POWER12 {
    typedef int8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef uint8_t OffsetBType;
    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{16, 256, 384};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{16, 256, 256};
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_POWER12::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_POWER12::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_POWER12::PackedStrides;

template <>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_POWER12>(
    int32_t ZeroPointA,
    bool AIsSigned
)
{
    if (!AIsSigned) {
        ZeroPointA = MLAS_GEMM_QUANT_KERNEL_POWER12::OffsetAType(ZeroPointA ^ 0x80);
    }
    return ZeroPointA;
}

template <>
MLAS_FORCEINLINE
    int32_t
    MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_POWER12>(
        int32_t ZeroPointB,
        bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_POWER12::OffsetBType(ZeroPointB ^ 0x80);
    }
    return ZeroPointB;
}

// Pack A for 16x16 blocks - transforms A[M][K] to packed format for MMA+
template <typename Vtype, bool AIsSigned>
void
MlasGemmQuantCopyPackA16x16(
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedAType *D,
    const uint8_t *A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t *RowSumBuffer
)
{
    constexpr uint8_t Flip = (AIsSigned ? 0 : 0x80);
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(Flip));
    typedef __vector signed char vec_t;
    size_t M = CountM;
    size_t K = CountK;
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedAType *D_start = D;
    // Process 16 rows at a time for MMA+ 16x16 blocks
    // Pack format: For each K/4 iteration, pack 16 rows x 4 columns
    while (CountM >= 8) {
        size_t rows_in_block = (CountM >= 16) ? 16 : 8;
        const uint8_t *a = A;
        __vector signed int vsum[4] = {{0}, {0}, {0}, {0}};
        size_t y = CountK;  // Changed from 'k' to match POWER10 convention

        // Fast path: Process K in chunks of 16 (VSX optimized)
        while (y >= 16) {
            // ===== Process rows 0-3, all 16 columns =====
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            Vtype a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            Vtype a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            Vtype a4 = *reinterpret_cast<const Vtype *>(&a[lda * 3]);

            // Transpose using VSX intrinsics
            Vtype vx = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
            Vtype vx1 = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));
            Vtype vx2 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
            Vtype vx3 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));

            Vtype vx4 = vec_xxpermdi(vx, vx1, 0);   // rows 0-3, cols 0-3
            Vtype vx5 = vec_xxpermdi(vx2, vx3, 0);  // rows 0-3, cols 4-7
            Vtype vx6 = vec_xxpermdi(vx, vx1, 3);   // rows 0-3, cols 8-11
            Vtype vx7 = vec_xxpermdi(vx2, vx3, 3);  // rows 0-3, cols 12-15

            // Apply sign conversion and store
            vec_t vxx = AIsSigned ? reinterpret_cast<vec_t>(vx4) : reinterpret_cast<vec_t>(vec_sub(vx4, vmask));
            vsum[0] = vec_sum4s(vxx, vsum[0]);
            *reinterpret_cast<vec_t *>(&D[0]) = vxx;

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx5) : reinterpret_cast<vec_t>(vec_sub(vx5, vmask));
            vsum[0] = vec_sum4s(vxx, vsum[0]);
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 4]) = vxx;

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx6) : reinterpret_cast<vec_t>(vec_sub(vx6, vmask));
            vsum[0] = vec_sum4s(vxx, vsum[0]);
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 8]) = vxx;

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx7) : reinterpret_cast<vec_t>(vec_sub(vx7, vmask));
            vsum[0] = vec_sum4s(vxx, vsum[0]);
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 12]) = vxx;

            // ===== Process rows 4-7, all 16 columns =====
            a1 = *reinterpret_cast<const Vtype *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const Vtype *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const Vtype *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const Vtype *>(&a[lda * 7]);

            vx = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
            vx1 = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));
            vx2 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
            vx3 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));

            Vtype vx8 = vec_xxpermdi(vx, vx1, 0);
            Vtype vx9 = vec_xxpermdi(vx2, vx3, 0);
            Vtype vx10 = vec_xxpermdi(vx, vx1, 3);
            Vtype vx11 = vec_xxpermdi(vx2, vx3, 3);

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx8) : reinterpret_cast<vec_t>(vec_sub(vx8, vmask));
            *reinterpret_cast<vec_t *>(&D[16]) = vxx;  // every row ,16 values - 16 bytes
            vsum[1] = vec_sum4s(vxx, vsum[1]);

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx9) : reinterpret_cast<vec_t>(vec_sub(vx9, vmask));
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 4 + 16]) = vxx;
            vsum[1] = vec_sum4s(vxx, vsum[1]);

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx10) : reinterpret_cast<vec_t>(vec_sub(vx10, vmask));
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 8 + 16]) = vxx;
            vsum[1] = vec_sum4s(vxx, vsum[1]);

            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx11) : reinterpret_cast<vec_t>(vec_sub(vx11, vmask));
            *reinterpret_cast<vec_t *>(&D[rows_in_block * 12 + 16]) = vxx;
            vsum[1] = vec_sum4s(vxx, vsum[1]);

            if (CountM >= 16) {
                // ===== Process rows 8-11, all 16 columns =====
                a1 = *reinterpret_cast<const Vtype *>(&a[lda * 8]);
                a2 = *reinterpret_cast<const Vtype *>(&a[lda * 9]);
                a3 = *reinterpret_cast<const Vtype *>(&a[lda * 10]);
                a4 = *reinterpret_cast<const Vtype *>(&a[lda * 11]);

                vx = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
                vx1 = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));
                vx2 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
                vx3 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));

                Vtype vx12 = vec_xxpermdi(vx, vx1, 0);
                Vtype vx13 = vec_xxpermdi(vx2, vx3, 0);
                Vtype vx14 = vec_xxpermdi(vx, vx1, 3);
                Vtype vx15 = vec_xxpermdi(vx2, vx3, 3);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx12) : reinterpret_cast<vec_t>(vec_sub(vx12, vmask));
                *reinterpret_cast<vec_t *>(&D[32]) = vxx;
                vsum[2] = vec_sum4s(vxx, vsum[2]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx13) : reinterpret_cast<vec_t>(vec_sub(vx13, vmask));
                *reinterpret_cast<vec_t *>(&D[96]) = vxx;
                vsum[2] = vec_sum4s(vxx, vsum[2]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx14) : reinterpret_cast<vec_t>(vec_sub(vx14, vmask));
                *reinterpret_cast<vec_t *>(&D[160]) = vxx;
                vsum[2] = vec_sum4s(vxx, vsum[2]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx15) : reinterpret_cast<vec_t>(vec_sub(vx15, vmask));
                *reinterpret_cast<vec_t *>(&D[224]) = vxx;
                vsum[2] = vec_sum4s(vxx, vsum[2]);

                // ===== Process rows 12-15, all 16 columns =====
                a1 = *reinterpret_cast<const Vtype *>(&a[lda * 12]);
                a2 = *reinterpret_cast<const Vtype *>(&a[lda * 13]);
                a3 = *reinterpret_cast<const Vtype *>(&a[lda * 14]);
                a4 = *reinterpret_cast<const Vtype *>(&a[lda * 15]);

                vx = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
                vx1 = reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));
                vx2 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1), reinterpret_cast<__vector int>(a2)));
                vx3 = reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3), reinterpret_cast<__vector int>(a4)));

                Vtype vx16 = vec_xxpermdi(vx, vx1, 0);
                Vtype vx17 = vec_xxpermdi(vx2, vx3, 0);
                Vtype vx18 = vec_xxpermdi(vx, vx1, 3);
                Vtype vx19 = vec_xxpermdi(vx2, vx3, 3);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx16) : reinterpret_cast<vec_t>(vec_sub(vx16, vmask));
                *reinterpret_cast<vec_t *>(&D[48]) = vxx;
                vsum[3] = vec_sum4s(vxx, vsum[3]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx17) : reinterpret_cast<vec_t>(vec_sub(vx17, vmask));
                *reinterpret_cast<vec_t *>(&D[112]) = vxx;
                vsum[3] = vec_sum4s(vxx, vsum[3]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx18) : reinterpret_cast<vec_t>(vec_sub(vx18, vmask));
                *reinterpret_cast<vec_t *>(&D[176]) = vxx;
                vsum[3] = vec_sum4s(vxx, vsum[3]);

                vxx = AIsSigned ? reinterpret_cast<vec_t>(vx19) : reinterpret_cast<vec_t>(vec_sub(vx19, vmask));
                *reinterpret_cast<vec_t *>(&D[240]) = vxx;
                vsum[3] = vec_sum4s(vxx, vsum[3]);

                D += 16 * 16;  // Advance by 256 bytes (16 rows × 16 bytes)
            } else
                D += 8 * 16;
            a += 16;  // Advance by 16 columns
            y -= 16;
        }
        // Process remaining K in chunks of 4 (handle all 16 rows)
        while (y >= 4) {
            // Load 4 columns from rows 0-3
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda]);
            int a3 = *reinterpret_cast<const int *>(&a[lda * 2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda * 3]);
            __vector int vx1 = {a1, a2, a3, a4};
            vec_t vx = AIsSigned ? reinterpret_cast<vec_t>(vx1) : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx1), vmask));
            vsum[0] = vec_sum4s(vx, vsum[0]);
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            // Load 4 columns from rows 4-7
            a1 = *reinterpret_cast<const int *>(&a[lda * 4]);
            a2 = *reinterpret_cast<const int *>(&a[lda * 5]);
            a3 = *reinterpret_cast<const int *>(&a[lda * 6]);
            a4 = *reinterpret_cast<const int *>(&a[lda * 7]);
            __vector int vx2 = {a1, a2, a3, a4};
            vx = AIsSigned ? reinterpret_cast<vec_t>(vx2) : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx2), vmask));
            vsum[1] = vec_sum4s(vx, vsum[1]);
            *reinterpret_cast<vec_t *>(&D[16]) = vx;

            if (rows_in_block >= 16) {
                // Load 4 columns from rows 8-11
                a1 = *reinterpret_cast<const int *>(&a[lda * 8]);
                a2 = *reinterpret_cast<const int *>(&a[lda * 9]);
                a3 = *reinterpret_cast<const int *>(&a[lda * 10]);
                a4 = *reinterpret_cast<const int *>(&a[lda * 11]);
                __vector int vx3 = {a1, a2, a3, a4};
                vx = AIsSigned ? reinterpret_cast<vec_t>(vx3) : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx3), vmask));
                vsum[2] = vec_sum4s(vx, vsum[2]);
                *reinterpret_cast<vec_t *>(&D[32]) = vx;

                // Load 4 columns from rows 12-15
                a1 = *reinterpret_cast<const int *>(&a[lda * 12]);
                a2 = *reinterpret_cast<const int *>(&a[lda * 13]);
                a3 = *reinterpret_cast<const int *>(&a[lda * 14]);
                a4 = *reinterpret_cast<const int *>(&a[lda * 15]);
                __vector int vx4 = {a1, a2, a3, a4};
                vx = AIsSigned ? reinterpret_cast<vec_t>(vx4) : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx4), vmask));
                vsum[3] = vec_sum4s(vx, vsum[3]);
                *reinterpret_cast<vec_t *>(&D[48]) = vx;
            }
            D += rows_in_block * 4;  // Advance by rows_in_block * 4 bytes
            a += 4;
            y -= 4;
        }

        // Handle remaining K (< 4)
        if (y > 0) {
            for (size_t row = 0; row < rows_in_block; row++) {
                vec_t v0 = {0};
                for (size_t i = 0; i < y; i++) {
                    v0[i] = a[row * lda + i];
                }
                v0 = AIsSigned ? v0 : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(v0), vmask));
                *reinterpret_cast<int32_t *>(D) = *reinterpret_cast<int32_t *>(&v0[0]);
                D += 4;

                size_t group = row / 4;
                vsum[group] = vec_sum4s(v0, vsum[group]);
            }
        }
        // RowSumBuffer
        size_t num_groups = (rows_in_block + 3) / 4;
        for (size_t group = 0; group < num_groups; group++) {
            // vsum[group] contains 4 separate row sums in elements [0],[1],[2],[3]
            for (size_t i = 0; i < 4; i++) {
                RowSumBuffer[group * 4 + i] = vsum[group][i];  // Use [i] not [3]!
            }
        }

        A += lda * rows_in_block;
        RowSumBuffer += rows_in_block;
        CountM -= rows_in_block;
    }

    // Handle remaining rows (< 16) - pad with zeros
    if (CountM > 0) {
        // Initialize vsum for each group of 4 rows (matching main loop)
        __vector signed int vsum[4] = {{0}, {0}, {0}, {0}};
        size_t k = CountK;
        const uint8_t *a = A;

        // Process K in chunks of 4 (matching main loop structure)
        while (k >= 4) {
            // Process in groups of 4 rows to match the main loop's structure
            for (size_t row_group = 0; row_group < 4; row_group++) {
                // Calculate which actual rows we're processing
                size_t row_start = row_group * 4;

                // Load 4 rows (or pad with zeros if beyond CountM)
                int a1 = (row_start + 0 < CountM) ? *reinterpret_cast<const int *>(&a[(row_start + 0) * lda]) : 0;
                int a2 = (row_start + 1 < CountM) ? *reinterpret_cast<const int *>(&a[(row_start + 1) * lda]) : 0;
                int a3 = (row_start + 2 < CountM) ? *reinterpret_cast<const int *>(&a[(row_start + 2) * lda]) : 0;
                int a4 = (row_start + 3 < CountM) ? *reinterpret_cast<const int *>(&a[(row_start + 3) * lda]) : 0;

                // Pack into vector (this creates the transpose)
                __vector int vx = {a1, a2, a3, a4};
                vec_t vxx = AIsSigned ? reinterpret_cast<vec_t>(vx) : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx), vmask));

                // Accumulate row sums for actual rows only
                if (row_start < CountM) {
                    vsum[row_group] = vec_sum4s(vxx, vsum[row_group]);
                }

                // Store the transposed data
                *reinterpret_cast<vec_t *>(&D[row_group * 16]) = vxx;
            }

            D += 64;  // Advance by rows_in_block * 4 bytes
            a += 4;   // Advance by 4 columns
            k -= 4;
        }

        // Handle remaining k < 4
        if (k > 0) {
            for (size_t row_group = 0; row_group < 4; row_group++) {
                size_t row_start = row_group * 4;

                // Load remaining columns (< 4) from each row
                vec_t v0 = {0};
                for (size_t i = 0; i < k; i++) {
                    if (row_start + 0 < CountM) v0[0] = a[(row_start + 0) * lda + i];
                    if (row_start + 1 < CountM) v0[1] = a[(row_start + 1) * lda + i];
                    if (row_start + 2 < CountM) v0[2] = a[(row_start + 2) * lda + i];
                    if (row_start + 3 < CountM) v0[3] = a[(row_start + 3) * lda + i];
                }

                v0 = AIsSigned ? v0 : reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(v0), vmask));

                // Accumulate row sums for actual rows only
                if (row_start < CountM) {
                    vsum[row_group] = vec_sum4s(v0, vsum[row_group]);
                }

                // Store the data
                *reinterpret_cast<int32_t *>(&D[row_group * 16]) = *reinterpret_cast<int32_t *>(&v0[0]);
            }

            D += 64;
        }

        // Store row sums - extract individual sums from each vsum
        for (size_t group = 0; group < 4; group++) {
            for (size_t i = 0; i < 4; i++) {
                size_t row = group * 4 + i;
                if (row < CountM) {
                    RowSumBuffer[row] = vsum[group][i];
                }
            }
        }
    }
}

template <>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_POWER12>(
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedAType *D,
    const uint8_t *A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t *RowSumBuffer,
    bool AIsSigned
)
{
    // For M < 8, use POWER10 packing which is optimized for small M
    if (CountM < 8) {
        MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_POWER10>(
            reinterpret_cast<MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType*>(D),
            A,
            lda,
            CountM,
            CountK,
            RowSumBuffer,
            AIsSigned
        );
        return;
    }
    
    // For M >= 8, use POWER12 16x16 packing
    if (AIsSigned) {
        MlasGemmQuantCopyPackA16x16<__vector signed char, true>(D, A, lda, CountM, CountK, RowSumBuffer);
    } else {
        MlasGemmQuantCopyPackA16x16<__vector unsigned char, false>(D, A, lda, CountM, CountK, RowSumBuffer);
    }
}

template <typename Vtype, bool BIsSigned>
void
MlasGemmQuantCopyPackB8x8(
    uint8_t *D,
    const uint8_t *B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t *ColumnSumBuffer
)
{
    [[maybe_unused]] constexpr uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    typedef __vector unsigned char vec_t;
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(BitFlipValue));
    vec_t mask = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of four, then the packed buffer
    // is padded with zero vectors.

    // Process 16 columns of matrix B in a loop.
    //
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedBType *D_start = D;
    size_t K = CountK;
    size_t N = CountN;
    size_t PackedK = ((CountK + 4 - 1) / 4) * 16;
    size_t k2 = PackedK;
    size_t k3 = PackedK * 2;
    size_t k4 = PackedK * 3;

    while (CountN >= 16) {
        // Process columns in groups of 4: cols 0-3, 4-7, 8-11, 12-15
        for (size_t col_group = 0; col_group < 16; col_group += 4) {
            const uint8_t *b_col = B + col_group;
            __vector unsigned int vsum = {0};
            size_t y = CountK;
            if (y >= 4) {
                do {
                    Vtype b1 = *reinterpret_cast<const Vtype *>(&b_col[0]);
                    Vtype b2 = *reinterpret_cast<const Vtype *>(&b_col[ldb]);
                    Vtype b3 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 2]);
                    Vtype b4 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 3]);
                    Vtype t1 = vec_mergeh(b1, b3);
                    Vtype t2 = vec_mergel(b1, b3);
                    Vtype t3 = vec_mergeh(b2, b4);
                    Vtype t4 = vec_mergel(b2, b4);
                    b1 = vec_mergeh(t1, t3);
                    b2 = vec_mergel(t1, t3);
                    b3 = vec_mergeh(t2, t4);
                    b4 = vec_mergel(t2, t4);
                    vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) : reinterpret_cast<vec_t>(b1);
                    vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) : reinterpret_cast<vec_t>(b2);
                    vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) : reinterpret_cast<vec_t>(b3);
                    vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) : reinterpret_cast<vec_t>(b4);

                    *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                    *reinterpret_cast<vec_t *>(&D[16]) = vx2;
                    *reinterpret_cast<vec_t *>(&D[32]) = vx3;
                    *reinterpret_cast<vec_t *>(&D[48]) = vx4;
                    vsum = vec_sum4s(vx1, vsum);
                    D += 64;
                    b_col += ldb * 4;
                    y -= 4;
                } while (y >= 4);
            }
            if (y >= 1) {
                Vtype b1 = *reinterpret_cast<const Vtype *>(&b_col[0]);
                Vtype b2 = (y >= 2) ? *reinterpret_cast<const Vtype *>(&b_col[ldb]) : vmask;
                Vtype b3 = (y >= 3) ? *reinterpret_cast<const Vtype *>(&b_col[ldb * 2]) : vmask;
                Vtype b4 = vmask;
                Vtype t1 = vec_mergeh(b1, b3);
                Vtype t2 = vec_mergel(b1, b3);
                Vtype t3 = vec_mergeh(b2, b4);
                Vtype t4 = vec_mergel(b2, b4);
                b1 = vec_mergeh(t1, t3);
                b2 = vec_mergel(t1, t3);
                b3 = vec_mergeh(t2, t4);
                b4 = vec_mergel(t2, t4);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) : reinterpret_cast<vec_t>(b1);
                vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) : reinterpret_cast<vec_t>(b2);
                vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) : reinterpret_cast<vec_t>(b3);
                vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) : reinterpret_cast<vec_t>(b4);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                *reinterpret_cast<vec_t *>(&D[16]) = vx2;
                *reinterpret_cast<vec_t *>(&D[32]) = vx3;
                *reinterpret_cast<vec_t *>(&D[48]) = vx4;
                vsum = vec_sum4s(vx1, vsum);
                D += 64;
            }
            // Store column sums for this 4-column group
            *ColumnSumBuffer++ = vsum[0];
            *ColumnSumBuffer++ = vsum[1];
            *ColumnSumBuffer++ = vsum[2];
            *ColumnSumBuffer++ = vsum[3];
        }  // End of col_group loop

        B += 16;
        CountN -= 16;
    }

    if (CountN >= 12) {
        // Process columns in groups of 4: cols 0-3, 4-7, 8-11
        for (size_t col_group = 0; col_group < 12; col_group += 4) {
            const uint8_t *b_col = B + col_group;
            __vector unsigned int vsum = {0};
            size_t y = CountK;
            if (y >= 4) {
                do {
                    Vtype b1 = *reinterpret_cast<const Vtype *>(&b_col[0]);
                    Vtype b2 = *reinterpret_cast<const Vtype *>(&b_col[ldb]);
                    Vtype b3 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 2]);
                    Vtype b4 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 3]);
                    Vtype t1 = vec_mergeh(b1, b3);
                    Vtype t2 = vec_mergel(b1, b3);
                    Vtype t3 = vec_mergeh(b2, b4);
                    Vtype t4 = vec_mergel(b2, b4);
                    b1 = vec_mergeh(t1, t3);
                    b2 = vec_mergel(t1, t3);
                    b3 = vec_mergeh(t2, t4);
                    b4 = vec_mergel(t2, t4);
                    vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) : reinterpret_cast<vec_t>(b1);
                    vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) : reinterpret_cast<vec_t>(b2);
                    vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) : reinterpret_cast<vec_t>(b3);
                    vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) : reinterpret_cast<vec_t>(b4);

                    *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                    *reinterpret_cast<vec_t *>(&D[16]) = vx2;
                    *reinterpret_cast<vec_t *>(&D[32]) = vx3;
                    *reinterpret_cast<vec_t *>(&D[48]) = vx4;
                    vsum = vec_sum4s(vx1, vsum);
                    D += 64;
                    b_col += ldb * 4;
                    y -= 4;
                } while (y >= 4);
            }
            *ColumnSumBuffer++ = vsum[0];
            *ColumnSumBuffer++ = vsum[1];
            *ColumnSumBuffer++ = vsum[2];
            *ColumnSumBuffer++ = vsum[3];
        }  // End of col_group loop

        B += 12;
        CountN -= 12;
    }
    if (CountN >= 8) {
        for (size_t col_group = 0; col_group < 12; col_group += 4) {
            const uint8_t *b_col = B + col_group;
            __vector unsigned int vsum = {0};
            size_t y = CountK;
            if (y >= 4) {
                do {
                    Vtype b1 = *reinterpret_cast<const Vtype *>(&b_col[0]);
                    Vtype b2 = *reinterpret_cast<const Vtype *>(&b_col[ldb]);
                    Vtype b3 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 2]);
                    Vtype b4 = *reinterpret_cast<const Vtype *>(&b_col[ldb * 3]);
                    Vtype t1 = vec_mergeh(b1, b3);
                    Vtype t2 = vec_mergel(b1, b3);
                    Vtype t3 = vec_mergeh(b2, b4);
                    Vtype t4 = vec_mergel(b2, b4);
                    b1 = vec_mergeh(t1, t3);
                    b2 = vec_mergel(t1, t3);
                    b3 = vec_mergeh(t2, t4);
                    b4 = vec_mergel(t2, t4);
                    vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) : reinterpret_cast<vec_t>(b1);
                    vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) : reinterpret_cast<vec_t>(b2);
                    vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) : reinterpret_cast<vec_t>(b3);
                    vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) : reinterpret_cast<vec_t>(b4);

                    *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                    *reinterpret_cast<vec_t *>(&D[16]) = vx2;
                    *reinterpret_cast<vec_t *>(&D[32]) = vx3;
                    *reinterpret_cast<vec_t *>(&D[48]) = vx4;
                    vsum = vec_sum4s(vx1, vsum);
                    D += 64;
                    b_col += ldb * 4;
                    y -= 4;
                } while (y >= 4);
            }
            *ColumnSumBuffer++ = vsum[0];
            *ColumnSumBuffer++ = vsum[1];
            *ColumnSumBuffer++ = vsum[2];
            *ColumnSumBuffer++ = vsum[3];
        }  // End of col_group loop

        B += 8;
        CountN -= 8;
    }

    // Process four columns of matrix B in a loop.
    //
    while (CountN >= 4) {
        const uint8_t *b = B;
        __vector unsigned int vsum = {0};
        size_t y = CountK;
        if (y >= 4) {
            do {
                int b1 = *reinterpret_cast<const int *>(&b[0]);
                int b2 = *reinterpret_cast<const int *>(&b[ldb]);
                int b3 = *reinterpret_cast<const int *>(&b[ldb * 2]);
                int b4 = *reinterpret_cast<const int *>(&b[ldb * 3]);
                __vector int vb = {b1, b2, b3, b4};
                Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) : reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s(vx1, vsum);
                D += 16;
                b += ldb * 4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = vmask;
            __vector int vb1 = reinterpret_cast<__vector int>(vb);
            vb1[0] = *reinterpret_cast<const int *>(&b[0]);
            if (y >= 2) {
                vb1[1] = *reinterpret_cast<const int *>(&b[ldb]);
            }
            if (y >= 3) {
                vb1[2] = *reinterpret_cast<const int *>(&b[ldb * 2]);
            }
            Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb1), reinterpret_cast<Vtype>(vb1), mask);
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) : reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s(vx1, vsum);
            D += 16;
        }
        *ColumnSumBuffer++ = vsum[0];
        *ColumnSumBuffer++ = vsum[1];
        *ColumnSumBuffer++ = vsum[2];
        *ColumnSumBuffer++ = vsum[3];
        B += 4;
        CountN -= 4;
    }

    //
    // Process the remaining columns of matrix B.
    //
    if (CountN > 0) {
        __vector unsigned int vsum = {0};
        const uint8_t *b = B;
        size_t y = CountK;
        if (y >= 4) {
            do {
                Vtype vb = vmask;
                if (CountN == 1) {
                    vb[0] = b[0];
                    vb[4] = b[ldb];
                    vb[8] = b[ldb * 2];
                    vb[12] = b[ldb * 3];
                }
                if (CountN == 2) {
                    vb[0] = b[0];
                    vb[1] = b[1];
                    vb[4] = b[ldb];
                    vb[5] = b[ldb + 1];
                    vb[8] = b[ldb * 2];
                    vb[9] = b[ldb * 2 + 1];
                    vb[12] = b[ldb * 3];
                    vb[13] = b[ldb * 3 + 1];
                }
                if (CountN == 3) {
                    vb[0] = b[0];
                    vb[1] = b[1];
                    vb[2] = b[2];
                    vb[4] = b[ldb];
                    vb[5] = b[ldb + 1];
                    vb[6] = b[ldb + 2];
                    vb[8] = b[ldb * 2];
                    vb[9] = b[ldb * 2 + 1];
                    vb[10] = b[ldb * 2 + 2];
                    vb[12] = b[ldb * 3];
                    vb[13] = b[ldb * 3 + 1];
                    vb[14] = b[ldb * 3 + 2];
                }
                Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) : reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s(vx1, vsum);
                D += 16;
                b += ldb * 4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = vmask;
            if (CountN == 1) {
                vb[0] = b[0];
                if (y >= 2) {
                    vb[4] = b[ldb];
                }
                if (y >= 3) {
                    vb[8] = b[ldb * 2];
                }
            }
            if (CountN == 2) {
                vb[0] = b[0];
                vb[1] = b[1];
                if (y >= 2) {
                    vb[4] = b[ldb];
                    vb[5] = b[ldb + 1];
                }
                if (y >= 3) {
                    vb[8] = b[ldb * 2];
                    vb[9] = b[ldb * 2 + 1];
                }
            }
            if (CountN == 3) {
                vb[0] = b[0];
                vb[1] = b[1];
                vb[2] = b[2];
                if (y >= 2) {
                    vb[4] = b[ldb];
                    vb[5] = b[ldb + 1];
                    vb[6] = b[ldb + 2];
                }
                if (y >= 3) {
                    vb[8] = b[ldb * 2];
                    vb[9] = b[ldb * 2 + 1];
                    vb[10] = b[ldb * 2 + 2];
                }
            }
            Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) : reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s(vx1, vsum);
            D += 16;
        }
        *ColumnSumBuffer++ = vsum[0];
        if (CountN >= 2) {
            *ColumnSumBuffer++ = vsum[1];
        }
        if (CountN >= 3) {
            *ColumnSumBuffer++ = vsum[2];
        }
    }
}

// POWER12 PackB - uses 8x8 column packing (same format as POWER10)
template <>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER12>(
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedBType *D,
    const uint8_t *B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t *ColumnSumBuffer,
    bool BIsSigned
)
{
    // Note: PackB doesn't have CountM parameter, so we can't conditionally
    // delegate based on M < 8. The M < 8 optimization is handled through:
    // 1. PackA delegation (which has CountM)
    // 2. Kernel routing (which checks CountM at runtime)
    // Both POWER10 and POWER12 use the same 8x8 column packing format.
    
    if (BIsSigned) {
        MlasGemmQuantCopyPackB8x8<__vector signed char, true>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    } else {
        MlasGemmQuantCopyPackB8x8<__vector unsigned char, false>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    }
}

// Type definitions for POWER12 SIMD vectors
typedef int32_t vec_i32 __attribute__((vector_size(16)));
// Helper function to store results with vectorized compensation for POWER12
// Stores one 8-row x 4-column block from a disassembled DMR1024
template <int VectorCount>
inline void
MlasQgemmStoreVectorMMAPlus(
    vec_i32 result[8],
    int32_t *C,
    size_t ldc,
    size_t row_count,
    bool ZeroMode,
    const int32_t *RowSumBuffer,
    const int32_t *ColumnSumBuffer,
    const int32_t *ZeroPointB,
    int col_offset
)
{
    // Load column sums as a vector (4 elements at a time)
    vec_i32 columnsum = *reinterpret_cast<const vec_i32 *>(&ColumnSumBuffer[col_offset]);

    if (ZeroPointB != nullptr) {
        // Load zero points as a vector
        vec_i32 zeropoint = *reinterpret_cast<const vec_i32 *>(&ZeroPointB[col_offset]);

        if (ZeroMode) {
            // Process each row in the 8-row block
            // result[7] = row 0, result[6] = row 1, ..., result[0] = row 7
            for (size_t row = 0; row < row_count && row < 8; row++) {
                vec_i32 vsum = vec_splats(RowSumBuffer[row]) * zeropoint + columnsum;
                *reinterpret_cast<vec_i32 *>(&C[row * ldc + col_offset]) = result[7 - row] + vsum;
            }
        } else {
            // Accumulate mode
            for (size_t row = 0; row < row_count && row < 8; row++) {
                vec_i32 vsum = vec_splats(RowSumBuffer[row]) * zeropoint + columnsum;
                *reinterpret_cast<vec_i32 *>(&C[row * ldc + col_offset]) += result[7 - row] + vsum;
            }
        }
    } else {
        // No zero point - simpler compensation
        if (ZeroMode) {
            for (size_t row = 0; row < row_count && row < 8; row++) {
                vec_i32 vsum = vec_splats(RowSumBuffer[row]) + columnsum;
                *reinterpret_cast<vec_i32 *>(&C[row * ldc + col_offset]) = result[7 - row] + vsum;
            }
        } else {
            for (size_t row = 0; row < row_count && row < 8; row++) {
                vec_i32 vsum = vec_splats(RowSumBuffer[row]) + columnsum;
                *reinterpret_cast<vec_i32 *>(&C[row * ldc + col_offset]) += result[7 - row] + vsum;
            }
        }
    }
}

// 16x16 MMA+ kernel implementation
template <>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_POWER12>(
    const MLAS_GEMM_QUANT_KERNEL_POWER12::PackedAType *A,
    const MLAS_GEMM_QUANT_KERNEL_POWER12::PackedBType *B,
    int32_t *C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t *RowSumBuffer,
    const int32_t *ColumnSumBuffer,
    const int32_t *ZeroPointB,
    bool ZeroMode
)
{
    // For M < 8, delegate to POWER10 kernel which handles small M efficiently
    if (CountM < 8) {
        return MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_POWER10>(
            reinterpret_cast<const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType*>(A),
            reinterpret_cast<const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType*>(B),
            C,
            PackedCountK,
            CountM,
            CountN,
            ldc,
            RowSumBuffer,
            ColumnSumBuffer,
            ZeroPointB,
            ZeroMode
        );
    }

    typedef __vector unsigned char vec_t;
    typedef int32_t vec_i32 __attribute__((vector_size(16)));

    size_t RowsHandled = 0;
    // Process 8 or 16 rows at a time using MMA+ blocks
    while (CountM >= 8) {
        size_t rows_in_block = (CountM >= 16) ? 16 : 8;
        size_t n = CountN;
        const int8_t *a = A;
        const uint8_t *b = B;
        int32_t *c = C;
        const int32_t *col_sum = ColumnSumBuffer;
        const int32_t *zp_b = ZeroPointB;

        // Process 16 columns at a time
        while (n > 0) {
            // Initialize 8 DMR1024 accumulators for 16x16 output
            __dmr1024 acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
            __builtin_mma_dmsetdmrz(&acc_0);
            __builtin_mma_dmsetdmrz(&acc_1);
            __builtin_mma_dmsetdmrz(&acc_2);
            __builtin_mma_dmsetdmrz(&acc_3);
            __builtin_mma_dmsetdmrz(&acc_4);
            __builtin_mma_dmsetdmrz(&acc_5);
            __builtin_mma_dmsetdmrz(&acc_6);
            __builtin_mma_dmsetdmrz(&acc_7);

            const int8_t *a_ptr = a;
            const uint8_t *b_ptr = b;

            // Accumulate over K dimension
            if (n >= 16) {
                for (size_t k = 0; k < PackedCountK; k++) {
                    // Load A: 16 rows x 4 columns (64 bytes = 2 vector pairs)
                    __vector_pair vec_A = __builtin_vsx_lxvp(0, (__vector_pair *)(a_ptr));
                    __vector_pair vec_A_1;
                    if (rows_in_block == 16) {
                        vec_A_1 = __builtin_vsx_lxvp(32, (__vector_pair *)(a_ptr));
                    }

                    // Load B: 4 rows x 16 columns (64 bytes = 4 vectors)
                    __vector_pair vec_B = __builtin_vsx_lxvp(0, (__vector_pair *)(b_ptr));
                    vec_t vecB0 = ((vec_t *)&vec_B)[0];
                    vec_t vecB1 = ((vec_t *)&vec_B)[1];

                    __vector_pair vec_B_1 = __builtin_vsx_lxvp(32, (__vector_pair *)(b_ptr));
                    vec_t vecB2 = ((vec_t *)&vec_B_1)[0];
                    vec_t vecB3 = ((vec_t *)&vec_B_1)[1];

                    // MMA+ operations: each produces 8x4 output
                    // Rows 0-7
                    __builtin_mma_dmxvi8gerx4pp(&acc_0, vec_A, vecB0);  // rows 0-7, cols 0-3
                    __builtin_mma_dmxvi8gerx4pp(&acc_1, vec_A, vecB1);  // rows 0-7, cols 4-7
                    __builtin_mma_dmxvi8gerx4pp(&acc_2, vec_A, vecB2);  // rows 0-7, cols 8-11
                    __builtin_mma_dmxvi8gerx4pp(&acc_3, vec_A, vecB3);  // rows 0-7, cols 12-15

                    // Rows 8-15 (only if rows_in_block == 16)
                    if (rows_in_block == 16) {
                        __builtin_mma_dmxvi8gerx4pp(&acc_4, vec_A_1, vecB0);  // rows 8-15, cols 0-3
                        __builtin_mma_dmxvi8gerx4pp(&acc_5, vec_A_1, vecB1);  // rows 8-15, cols 4-7
                        __builtin_mma_dmxvi8gerx4pp(&acc_6, vec_A_1, vecB2);  // rows 8-15, cols 8-11
                        __builtin_mma_dmxvi8gerx4pp(&acc_7, vec_A_1, vecB3);  // rows 8-15, cols 12-15
                    }

                    a_ptr += rows_in_block * 4;  // rows_in_block rows * 4 cols
                    b_ptr += 64;                 // 4 rows * 16 cols
                }
                // Disassemble and store results
                vec_i32 result[8];

                // Process each accumulator and store to C
                __dmr1024 *accs[8] = {&acc_0, &acc_1, &acc_2, &acc_3, &acc_4, &acc_5, &acc_6, &acc_7};
                int row_offsets[8] = {0, 0, 0, 0, 8, 8, 8, 8};
                int col_offsets[8] = {0, 4, 8, 12, 0, 4, 8, 12};

                size_t num_accs = (rows_in_block == 16) ? 8 : 4;
                for (size_t acc_idx = 0; acc_idx < num_accs; acc_idx++) {
                    __builtin_mma_disassemble_dmr(result, accs[acc_idx]);
                    int row_base = row_offsets[acc_idx];
                    int col_base = col_offsets[acc_idx];

                    // Store 8 rows x 4 columns (results are in reverse order: 7->0)
                    for (int i = 7; i >= 0; i--) {
                        size_t row = row_base + (7 - i);
                        if (row >= rows_in_block) continue;  // Skip if beyond actual rows
                        vec_i32 *rowC = (vec_i32 *)&c[row * ldc + col_base];
                        if (ZeroMode) {
                            rowC[0] = result[i];
                        } else {
                            rowC[0] = vec_add(rowC[0], result[i]);
                        }
                    }
                }

                if (zp_b) {
                    for (size_t row = 0; row < rows_in_block; row++) {
                        for (size_t col = 0; col < 16; col++) {
                            c[row * ldc + col] += RowSumBuffer[row] * zp_b[col];
                            c[row * ldc + col] += col_sum[col];
                        }
                    }
                } else {
                    // When ZeroPointB is NULL, add RowSumBuffer directly
                    for (size_t row = 0; row < rows_in_block; row++) {
                        for (size_t col = 0; col < 16; col++) {
                            c[row * ldc + col] += RowSumBuffer[row];
                            c[row * ldc + col] += col_sum[col];
                        }
                    }
                }

                c += 16;
                b += PackedCountK * 64;
                col_sum += 16;
                if (zp_b) zp_b += 16;
                n -= 16;
            }
            // Handle N >= 12 (12 columns) - Use 6 accumulators
            else {
                if (n >= 12) {
                    for (size_t k = 0; k < PackedCountK; k++) {
                        __vector_pair vec_A = __builtin_vsx_lxvp(0, (__vector_pair *)(a_ptr));
                        __vector_pair vec_A_1;
                        if (rows_in_block == 16) {
                            vec_A_1 = __builtin_vsx_lxvp(32, (__vector_pair *)(a_ptr));
                        }

                        __vector_pair vec_B = __builtin_vsx_lxvp(0, (__vector_pair *)(b_ptr));
                        vec_t vecB0 = ((vec_t *)&vec_B)[0];
                        vec_t vecB1 = ((vec_t *)&vec_B)[1];

                        vec_t vecB2 = *((vec_t *)(b_ptr + 32));
                        __builtin_mma_dmxvi8gerx4pp(&acc_0, vec_A, vecB0);  // rows 0-7, cols 0-3
                        __builtin_mma_dmxvi8gerx4pp(&acc_1, vec_A, vecB1);  // rows 0-7, cols 4-7
                        __builtin_mma_dmxvi8gerx4pp(&acc_2, vec_A, vecB2);  // rows 0-7, cols 8-11

                        if (rows_in_block == 16) {
                            __builtin_mma_dmxvi8gerx4pp(&acc_4, vec_A_1, vecB0);  // rows 8-15, cols 0-3
                            __builtin_mma_dmxvi8gerx4pp(&acc_5, vec_A_1, vecB1);  // rows 8-15, cols 4-7
                            __builtin_mma_dmxvi8gerx4pp(&acc_6, vec_A_1, vecB2);  // rows 8-15, cols 8-11
                        }

                        a_ptr += rows_in_block * 4;  // rows_in_block rows * 4 cols
                        b_ptr += 64;                 // 4 rows * 16 cols
                    }
                    vec_i32 result[8];
                    __dmr1024 *accs[6] = {&acc_0, &acc_1, &acc_2, &acc_4, &acc_5, &acc_6};
                    int row_offsets[6] = {0, 0, 0, 8, 8, 8};
                    int col_offsets[6] = {0, 4, 8, 0, 4, 8};

                    size_t num_accs = (rows_in_block == 16) ? 6 : 3;

                    for (size_t acc_idx = 0; acc_idx < num_accs; acc_idx++) {
                        __builtin_mma_disassemble_dmr(result, accs[acc_idx]);
                        int row_base = row_offsets[acc_idx];
                        int col_base = col_offsets[acc_idx];

                        // Determine how many rows to process for this accumulator
                        size_t rows_to_process = (row_base == 0) ? std::min(rows_in_block, size_t(8)) : (rows_in_block - 8);

                        // Use vectorized compensation for each 8-row x 4-column block
                        MlasQgemmStoreVectorMMAPlus<4>(result, &c[row_base * ldc], ldc, rows_to_process, ZeroMode, &RowSumBuffer[row_base], col_sum, zp_b, col_base);
                    }
                    c += 12;
                    b += PackedCountK * 48;
                    col_sum += 12;
                    if (zp_b) zp_b += 12;
                    n -= 12;
                }
                // Handle N >= 8 (8 columns) - Use 4 accumulators
                if (n >= 8) {
                    for (size_t k = 0; k < PackedCountK; k++) {
                        __vector_pair vec_A = __builtin_vsx_lxvp(0, (__vector_pair *)(a_ptr));
                        __vector_pair vec_A_1;
                        if (rows_in_block == 16) {
                            vec_A_1 = __builtin_vsx_lxvp(32, (__vector_pair *)(a_ptr));
                        }

                        __vector_pair vec_B = __builtin_vsx_lxvp(0, (__vector_pair *)(b_ptr));
                        vec_t vecB0 = ((vec_t *)&vec_B)[0];
                        vec_t vecB1 = ((vec_t *)&vec_B)[1];

                        __builtin_mma_dmxvi8gerx4pp(&acc_0, vec_A, vecB0);  // rows 0-7, cols 0-3
                        __builtin_mma_dmxvi8gerx4pp(&acc_1, vec_A, vecB1);  // rows 0-7, cols 4-7

                        if (rows_in_block == 16) {
                            __builtin_mma_dmxvi8gerx4pp(&acc_4, vec_A_1, vecB0);  // rows 8-15, cols 0-3
                            __builtin_mma_dmxvi8gerx4pp(&acc_5, vec_A_1, vecB1);  // rows 8-15, cols 4-7
                        }

                        a_ptr += rows_in_block * 4;  // rows_in_block rows * 4 cols
                        b_ptr += 64;                 // To work with B packing structure (redundant)
                    }
                    vec_i32 result[8];
                    __dmr1024 *accs[4] = {&acc_0, &acc_1, &acc_4, &acc_5};
                    int row_offsets[4] = {0, 0, 8, 8};
                    int col_offsets[4] = {0, 4, 0, 4};

                    size_t num_accs = (rows_in_block == 16) ? 4 : 2;

                    for (size_t acc_idx = 0; acc_idx < num_accs; acc_idx++) {
                        __builtin_mma_disassemble_dmr(result, accs[acc_idx]);
                        int row_base = row_offsets[acc_idx];
                        int col_base = col_offsets[acc_idx];

                        for (int i = 7; i >= 0; i--) {
                            size_t row = row_base + (7 - i);
                            if (row >= rows_in_block) continue;
                            vec_i32 *rowC = (vec_i32 *)&c[row * ldc + col_base];
                            if (ZeroMode) {
                                rowC[0] = result[i];
                            } else {
                                rowC[0] = vec_add(rowC[0], result[i]);
                            }
                        }
                    }

                    if (zp_b) {
                        for (size_t row = 0; row < rows_in_block; row++) {
                            for (size_t col = 0; col < 8; col++) {
                                c[row * ldc + col] += RowSumBuffer[row] * zp_b[col];
                                c[row * ldc + col] += col_sum[col];
                            }
                        }
                    } else {
                        for (size_t row = 0; row < rows_in_block; row++) {
                            for (size_t col = 0; col < 8; col++) {
                                c[row * ldc + col] += RowSumBuffer[row];
                                c[row * ldc + col] += col_sum[col];
                            }
                        }
                    }


                    c += 8;
                    b += PackedCountK * 32;
                    col_sum += 8;
                    if (zp_b) zp_b += 8;
                    n -= 8;
                }
                // Handle N >= 4 (4 columns) - Use 2 accumulators
                if (n >= 4) {
                    for (size_t k = 0; k < PackedCountK; k++) {
                        __vector_pair vec_A = __builtin_vsx_lxvp(0, (__vector_pair *)(a_ptr));
                        __vector_pair vec_A_1;
                        if (rows_in_block == 16) {
                            vec_A_1 = __builtin_vsx_lxvp(32, (__vector_pair *)(a_ptr));
                        }

                        vec_t vecB0 = *((vec_t *)(b_ptr));
                        __builtin_mma_dmxvi8gerx4pp(&acc_0, vec_A, vecB0);  // rows 0-7, cols 0-3

                        if (rows_in_block == 16) {
                            __builtin_mma_dmxvi8gerx4pp(&acc_4, vec_A_1, vecB0);  // rows 8-15, cols 0-3
                        }

                        a_ptr += rows_in_block * 4;  // rows_in_block rows * 4 cols
                        b_ptr += 16;                 // 4 rows * 16 cols
                    }

                    vec_i32 result[8];
                    __dmr1024 *accs[2] = {&acc_0, &acc_4};
                    int row_offsets[2] = {0, 8};
                    int col_offsets[2] = {0, 0};

                    size_t num_accs = (rows_in_block == 16) ? 2 : 1;

                    for (size_t acc_idx = 0; acc_idx < num_accs; acc_idx++) {
                        __builtin_mma_disassemble_dmr(result, accs[acc_idx]);
                        int row_base = row_offsets[acc_idx];
                        int col_base = col_offsets[acc_idx];

                        for (int i = 7; i >= 0; i--) {
                            size_t row = row_base + (7 - i);
                            if (row >= rows_in_block) continue;
                            vec_i32 *rowC = (vec_i32 *)&c[row * ldc + col_base];
                            if (ZeroMode) {
                                rowC[0] = result[i];
                            } else {
                                rowC[0] = vec_add(rowC[0], result[i]);
                            }
                        }
                    }

                    if (zp_b) {
                        for (size_t row = 0; row < rows_in_block; row++) {
                            for (size_t col = 0; col < 4; col++) {
                                c[row * ldc + col] += RowSumBuffer[row] * zp_b[col];
                                c[row * ldc + col] += col_sum[col];
                            }
                        }
                    } else {
                        for (size_t row = 0; row < rows_in_block; row++) {
                            for (size_t col = 0; col < 4; col++) {
                                c[row * ldc + col] += RowSumBuffer[row];
                                c[row * ldc + col] += col_sum[col];
                            }
                        }
                    }

                    c += 4;
                    b += PackedCountK * 16;
                    col_sum += 4;
                    if (zp_b) zp_b += 4;
                    n -= 4;
                }
                // Handle remaining columns (n < 4) - scalar fallback
                while (n > 0) {
                    // Process one column at a time
                    for (size_t row = 0; row < rows_in_block; row++) {
                        int32_t sum = 0;
                        const int8_t *a_row = a + row * PackedCountK * 4;
                        const uint8_t *b_col = b;

                        // Compute dot product over K
                        for (size_t k = 0; k < PackedCountK * 4; k++) {
                            sum += (int32_t)a_row[k] * (int32_t)b_col[k];
                        }

                        // Zero-point compensation
                        if (zp_b) {
                            sum += RowSumBuffer[row] * zp_b[0];
                        } else {
                            sum += RowSumBuffer[row];
                        }
                        sum += col_sum[0];

                        // Store
                        if (ZeroMode) {
                            c[row * ldc] = sum;
                        } else {
                            c[row * ldc] += sum;
                        }
                    }

                    c++;
                    b += PackedCountK * 4;
                    col_sum++;
                    if (zp_b) zp_b++;
                    n--;
                }
            }
        }
        A += PackedCountK * rows_in_block * 4;
        C += rows_in_block * ldc;
        RowSumBuffer += rows_in_block;
        CountM -= rows_in_block;
        RowsHandled += rows_in_block;
    }
    // Handle remaining rows (< 16) - scalar fallback
    if (CountM > 0) {
        for (size_t row = 0; row < CountM; row++) {
            size_t n = CountN;
            const int8_t *a_ptr = A + row * PackedCountK * 4;
            const uint8_t *b_ptr = B;
            int32_t *c_ptr = C + row * ldc;
            const int32_t *col_sum = ColumnSumBuffer;
            const int32_t *zp_b = ZeroPointB;

            while (n > 0) {
                int32_t sum = 0;
                for (size_t k = 0; k < PackedCountK * 4; k++) {
                    sum += (int32_t)a_ptr[k] * (int32_t)b_ptr[k];
                }
                if (!ZeroMode) {
                    sum += *c_ptr;
                }
                sum += RowSumBuffer[row] * (zp_b ? zp_b[0] : 0);
                sum += col_sum[0];
                *c_ptr = sum;

                c_ptr++;
                b_ptr += PackedCountK * 4;
                col_sum++;
                if (zp_b) zp_b++;
                n--;
            }
        }
        RowsHandled += CountM;
    }

    return RowsHandled;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemm8X8DispatchPOWER12 = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_POWER12>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_QUANT_KERNEL_POWER12>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER12>,
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedK,
    MLAS_GEMM_QUANT_KERNEL_POWER12::PackedStrides.K,
    16  // Kernel M stride
};
