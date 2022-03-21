import "regent"

local blas = terralib.includecstring [[

extern void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

extern void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

extern void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);

extern void dgetri_(int *m, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);

extern void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, double *w, double *work, int *lwork, int *info);

]]

terralib.linklibrary("libmkl_core.so")
terralib.linklibrary("libmkl_sequential.so")
terralib.linklibrary("libmkl_intel_lp64.so")

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")

-- Import max/min for Terra
max = regentlib.fmax
min = regentlib.fmin

fspace csb_entry
{
    {rloc, cloc} : uint32,
    val: double,
}

--[[
    TERRA FUNCTIONS BEGIN HERE
]]--

terra get_raw_ptr(index: int, block_width: int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
    var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
    var rect : c.legion_rect_1d_t
    var subrect : c.legion_rect_1d_t
    var offsets : c.legion_byte_offset_t[1]

    rect.lo.x[0] = index * block_width
    rect.hi.x[0] = (index + 1) * block_width - 1

    return [&double](c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, &(offsets[0])))
end

terra get_raw_ptr_by_range(s: int, e: int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
    var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
    var rect : c.legion_rect_1d_t
    var subrect : c.legion_rect_1d_t
    var offsets : c.legion_byte_offset_t[1]

    rect.lo.x[0] = s
    rect.hi.x[0] = e - 1

    return [&double](c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, &(offsets[0])))
end
terra dgemm_terra(prA: c.legion_physical_region_t,
                  fldA: c.legion_field_id_t,
                  prB: c.legion_physical_region_t,
                  fldB: c.legion_field_id_t,
                  prC: c.legion_physical_region_t,
                  fldC: c.legion_field_id_t,
                  m: int, n: int, k: int,
                  alpha: double, beta: double)
    var ptr_A = get_raw_ptr(0, m*k, prA, fldA)
    var ptr_B = get_raw_ptr(0, k*n, prB, fldB)
    var ptr_C = get_raw_ptr(0, m*n, prC, fldC)

    blas.cblas_dgemm(101, 111, 111, m, n, k, alpha, ptr_A, k, ptr_B, n, beta, ptr_C, n)
end

terra dgemm_terra_part(prA: c.legion_physical_region_t,
                  fldA: c.legion_field_id_t,
                  prB: c.legion_physical_region_t,
                  fldB: c.legion_field_id_t,
                  prC: c.legion_physical_region_t,
                  fldC: c.legion_field_id_t,
                  m: int, n: int, k: int, part: int, M: int, 
                  alpha: double, beta: double)
    var m_defacto = min(m, M - part*m)
    var ptr_A = get_raw_ptr_by_range(m*k*part, m*k*part + m_defacto*k, prA, fldA)
    var ptr_B = get_raw_ptr_by_range(0, k*n, prB, fldB)
    var ptr_C = get_raw_ptr_by_range(m*n*part, m*n*part + m_defacto*n, prC, fldC)

    blas.cblas_dgemm(101, 111, 111, m_defacto, n, k, alpha, ptr_A, k, ptr_B, n, beta, ptr_C, n)
end
terra dgemm_offset_terra(prA: c.legion_physical_region_t,
                  fldA: c.legion_field_id_t,
                  prB: c.legion_physical_region_t,
                  fldB: c.legion_field_id_t,
                  bOffset: int,
                  prC: c.legion_physical_region_t,
                  fldC: c.legion_field_id_t,
                  m: int, n: int, k: int,
                  alpha: double, beta: double)
    var ptr_A = get_raw_ptr(0, m*k, prA, fldA)
    var ptr_B = get_raw_ptr(bOffset, k*n, prB, fldB)
    var ptr_C = get_raw_ptr(0, m*n, prC, fldC)

    blas.cblas_dgemm(101, 111, 111, m, n, k, alpha, ptr_A, k, ptr_B, n, beta, ptr_C, n)
end

terra dgemm_offset_terra_part(prA: c.legion_physical_region_t,
                  fldA: c.legion_field_id_t,
                  prB: c.legion_physical_region_t,
                  fldB: c.legion_field_id_t,
                  bOffset: int,
                  prC: c.legion_physical_region_t,
                  fldC: c.legion_field_id_t,
                  m: int, n: int, k: int, part: int, 
                  M: int, alpha: double, beta: double)
    var m_defacto = min(m, M - part*m)
    var ptr_A = get_raw_ptr_by_range(m*k*part, m*k*part + m_defacto*k, prA, fldA)
    var ptr_B = get_raw_ptr_by_range(bOffset, bOffset + k*n, prB, fldB)
    var ptr_C = get_raw_ptr_by_range(m*n*part, m*n*part + m_defacto*k, prC, fldC)

    blas.cblas_dgemm(101, 111, 111, m_defacto, n, k, alpha, ptr_A, k, ptr_B, n, beta, ptr_C, n)
end

terra dgemm_transpose_terra(prA: c.legion_physical_region_t,
                            fldA: c.legion_field_id_t,
                            prB: c.legion_physical_region_t,
                            fldB: c.legion_field_id_t,
                            prC: c.legion_physical_region_t,
                            fldC: c.legion_field_id_t,
                            m: int, n: int, k: int,
                            alpha: double, beta: double)
    var ptr_A = get_raw_ptr(0, m*k, prA, fldA)
    var ptr_B = get_raw_ptr(0, k*n, prB, fldB)
    var ptr_C = get_raw_ptr(0, m*n, prC, fldC)

    blas.cblas_dgemm(101, 112, 111, m, n, k, alpha, ptr_A, m, ptr_B, n, beta, ptr_C, n)
end

terra dgemm_transpose_terra_part(prA: c.legion_physical_region_t,
                            fldA: c.legion_field_id_t,
                            prB: c.legion_physical_region_t,
                            fldB: c.legion_field_id_t,
                            prC: c.legion_physical_region_t,
                            fldC: c.legion_field_id_t,
                            m: int, n: int, k: int, part: int, M: int,
                            alpha: double, beta: double)
    var k_defacto = min(k, M - part*k)
    var ptr_A = get_raw_ptr_by_range(m*k*part, m*k*part + m*k_defacto, prA, fldA)
    var ptr_B = get_raw_ptr_by_range(k*n*part, k*n*part + k_defacto*n, prB, fldB)
    var ptr_C = get_raw_ptr_by_range(0, m*n, prC, fldC)

    blas.cblas_dgemm(101, 112, 111, m, n, k_defacto, alpha, ptr_A, m, ptr_B, n, beta, ptr_C, n)
end

terra dpotrf_terra(prA: c.legion_physical_region_t,
                   fldA: c.legion_field_id_t,
                   m: int)
    var uplo : rawstring = 'U'
    var m_ : int[1]
    var ptr_A = get_raw_ptr(0, m*m, prA, fldA)
    var info : int[1]

    m_[0] = m
    blas.dpotrf_(uplo, m_, ptr_A, m_, info);

    if info[0] ~= 0 then
        c.printf("Error: Cholesky, info: %d\n", info[0])
    end
end

terra inverse_terra(a: &double, m: int)
    var ipiv : &int32 = [&int32](c.malloc((m + 1) * 4))
    var lda : int[1]
    var lwork : int[1]
    var info: int[1]
    var m_: int[1]
    var work_query: double[1]

    m_[0] = m
    lda[0] = m
    lwork[0] = -1

    blas.dgetrf_(m_, m_, a, lda, ipiv, info);
    if info[0] < 0 then
        c.printf("Error: LU, info: %d\n", info[0])
    end

    blas.dgetri_(m_, a, lda, ipiv, work_query, lwork, info)
    if info[0] < 0 then
        c.printf("Error: (LU)^-1 #1, info: %d\n", info[0])
    end

    lwork[0] = [int32](work_query[0])
    var work : &double = [&double](c.malloc(lwork[0] * 8))

    blas.dgetri_(m_, a, lda, ipiv, work, lwork, info)
    if info[0] < 0 then
        c.printf("Error: (LU)^-1 #2, info: %d\n", info[0])
    end

    c.free(work)
end

terra eigencomp_terra(prA: c.legion_physical_region_t,
                      fldA: c.legion_field_id_t,
                      prB: c.legion_physical_region_t,
                      fldB: c.legion_field_id_t,
                      prW: c.legion_physical_region_t,
                      fldW: c.legion_field_id_t,
                      blocksize: int)
    var itype : int[1]
    var jobz : rawstring = 'V'
    var uplo : rawstring = 'U'
    var blocksize_ : int[1]
    var work_query: double[1]
    var lwork : int[1]
    var info: int[1]
    var ptr_A = get_raw_ptr(0, blocksize * blocksize, prA, fldA)
    var ptr_B = get_raw_ptr(0, blocksize * blocksize, prB, fldB)
    var ptr_W = get_raw_ptr(0, blocksize, prW, fldW)

    itype[0] = 1;
    blocksize_[0] = blocksize
    lwork[0] = -1
    
    blas.dsygv_(itype, jobz, uplo, blocksize_, ptr_A, blocksize_, ptr_B, blocksize_, ptr_W, work_query, lwork, info)
    if info[0] ~= 0 then
        c.printf("Error: EIG #1, info: %d\n", info[0])
    end
    
    lwork[0] = [int32](work_query[0])
    var work : &double = [&double](c.malloc(lwork[0] * 8))
    blas.dsygv_(itype, jobz, uplo, blocksize_, ptr_A, blocksize_, ptr_B, blocksize_, ptr_W, work, lwork, info)
    if info[0] ~= 0 then
        c.printf("Error: EIG #2, info: %d\n", info[0])
    end

    c.free(work)
end                      


--[[
    REGENT TASKS BEGIN HERE
]]--


task setup_matrix(rA: region(ispace(int1d), csb_entry),
                  rP: region(ispace(int1d), int),
                  block_width: int32, num_subregions: int32,
                  graph : regentlib.string)
where reads  writes(rA, rP) do
    -- read numrows, numcols, *colptrs, *irem, *xrem
    var file = c.fopen([rawstring](graph), "rb")
    regentlib.assert(not isnull(file), "failed to open graph file")

    var numrows: int32
    var numcols: int32
    var nnonzero: uint32
    var nnzmalloc: uint64 = 4

    c.fread(&numrows, 4, 1, file)
    c.fread(&numcols, 4, 1, file)
    c.fread(&nnonzero, 4, 1, file)
    c.printf("numrows = %d -- numcols = %d -- nnz = %d\n", numrows, numcols, nnonzero)

    nnzmalloc = nnzmalloc * nnonzero

    var colptrs : &int32 = [&int32](c.malloc((numcols + 1) * 4))
    var irem : &int32 = [&int32](c.malloc(nnzmalloc))
    var xrem : &float = [&float](c.malloc(nnzmalloc))

    c.fread(colptrs, 4, numcols + 1, file);
    c.fread(irem, 4, nnonzero, file);
    c.fread(xrem, 4, nnonzero, file);
    c.printf("colptrs, irem, xrem read\n");

    for i = 0, numcols + 1 do
        colptrs[i] = colptrs[i] - 1
    end

    for i = 0, nnonzero do
        irem[i] = irem[i] - 1
    end

    -- convert it to flattened csb entries
    var nnz_block : &int32 = [&int32](c.malloc(num_subregions * num_subregions * 4))
    var blkr : int32
    var blkc : int32
    
    c.printf("allocated space for nnz counts per block\n")

    for i = 0, num_subregions * num_subregions do
        nnz_block[i] = 0
    end

    c.printf("nnz counts per block initialized\n")
    
    for i = 0, numcols do
        for j = colptrs[i], colptrs[i+1] do
            blkr = irem[j]/block_width
            blkc = i/block_width
            nnz_block[blkr * num_subregions + blkc] = nnz_block[blkr * num_subregions + blkc] + 1
        end
    end

    rP[0] = 0
    for i = 0, num_subregions * num_subregions do
        rP[i + 1] = rP[i] + nnz_block[i]
    end

    var prev = nnz_block[0]
    var tmp = 0
    nnz_block[0] = 0
    for i = 1, num_subregions * num_subregions do
        tmp = prev
        prev = nnz_block[i]
        nnz_block[i] = nnz_block[i-1] + tmp
    end

    if nnz_block[num_subregions * num_subregions - 1] + prev ~= nnonzero then
        c.printf("nnz_block not computed correctly")
    end

    c.printf("nnz counts per block computed\n")

    var blk = 0
    var pos = 0
    for i = 0, numcols do
        for j = colptrs[i], colptrs[i+1] do
            blk = (irem[j]/block_width) * num_subregions + i/block_width
            pos = nnz_block[blk]
            
            rA[pos].rloc = irem[j]%block_width
            rA[pos].cloc = i%block_width
            --rA[pos].rloc = irem[j]
            --rA[pos].cloc = i
            rA[pos].val = xrem[j]

            nnz_block[blk] = nnz_block[blk] + 1
        end
    end

    c.printf("done with matrix setup\n")

    c.fclose(file)
    c.free(colptrs)
    c.free(irem)
    c.free(xrem)
    c.free(nnz_block)
end

task GetRandomVectors(rVB: region(ispace(int1d), double),
                      m: int, blocksize: int)
where writes(rVB) do    
    std.srand(0)
    for i = 0, m * blocksize do
        rVB[i] = [double](std.rand())/[double](std.RAND_MAX)
        --rVB[i] = -1.00 + std.rand() % 2
    end
end                      

task Reset(rA: region(ispace(int1d), double),
           m: int, n: int)
where writes(rA) do
    for i = 0, m do
        for j = 0, n do
            rA[i*n + j] = 0.0
        end
    end
end

task Reset_part(rA: region(ispace(int1d), double),
                m: int, n: int, part: int, M: int)
where writes(rA) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rA[i*n + j] = 0.0
        end
    end
end

task Identity(rA: region(ispace(int1d), double),
              m: int)
where writes(rA) do
    for i = 0, m do
        for j = 0, m do
            if i == j then
                rA[i*m + j] = 1.0
            else
                rA[i*m + j] = 0.0
            end
        end
    end
end

task XY(rA: region(ispace(int1d), double),
         rB: region(ispace(int1d), double),
         rC: region(ispace(int1d), double),
         m: int, n: int, k: int,
         alpha: double, beta: double)
where reads(rA, rB), writes(rC) do
    dgemm_terra(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, alpha, beta)
end

task XY_part(rA: region(ispace(int1d), double),
         rB: region(ispace(int1d), double),
         rC: region(ispace(int1d), double),
         m: int, n: int, k: int, part: int, M: int,
         alpha: double, beta: double)
where reads(rA, rB), writes(rC) do
    dgemm_terra_part(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, part, M, alpha, beta)
end

task XYOffset(rA: region(ispace(int1d), double),
         rB: region(ispace(int1d), double),
         bOffset: int,
         rC: region(ispace(int1d), double),
         m: int, n: int, k: int,
         alpha: double, beta: double)
where reads(rA, rB), reads writes(rC) do
    dgemm_offset_terra(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          bOffset,
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, alpha, beta)
end

task XYOffset_part(rA: region(ispace(int1d), double),
                   rB: region(ispace(int1d), double),
                   bOffset: int,
                   rC: region(ispace(int1d), double),
                   m: int, n: int, k: int, part: int, 
                   M: int, alpha: double, beta: double)
where reads(rA, rB), reads writes(rC) do
    dgemm_offset_terra_part(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          bOffset,
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, part, M, alpha, beta)
end

task XTY(rA: region(ispace(int1d), double),
         rB: region(ispace(int1d), double),
         rC: region(ispace(int1d), double),
         m: int, n: int, k: int,
         alpha: double, beta: double)
where reads(rA, rB), reads writes(rC) do
    dgemm_transpose_terra(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, alpha, beta)
end

task XTY_part(rA: region(ispace(int1d), double),
              rB: region(ispace(int1d), double),
              rC: region(ispace(int1d), double),
              m: int, n: int, k: int, part: int, M: int,
              alpha: double, beta: double)
where reads(rA, rB), reduces+(rC) do
    dgemm_transpose_terra_part(__physical(rA)[0], __fields(rA)[0],
                          __physical(rB)[0], __fields(rB)[0],
                          __physical(rC)[0], __fields(rC)[0],
                          m, n, k, part, M, alpha, beta)
end

task Transpose(rSrc: region(ispace(int1d), double),
               rDst: region(ispace(int1d), double),
               m: int, n: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[j*m + i] = rSrc[i*n + j]
        end
    end
end

task Cholesky(rA: region(ispace(int1d), double),
              m: int)
where reads writes(rA) do
    dpotrf_terra(__physical(rA)[0], __fields(rA)[0], m)
end

task ResetBelowDiagonal(rA: region(ispace(int1d), double),
                        m: int)
where writes(rA) do
    for i = 0, m do
        for j = 0, i do
            rA[i*m + j] = 0.0
        end
    end
end                        

task Copy(rSrc: region(ispace(int1d), double),
          rDst: region(ispace(int1d), double),
          m: int, n: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j]
        end
    end
end          

task Copy_part(rSrc: region(ispace(int1d), double),
          rDst: region(ispace(int1d), double),
          m: int, n: int, part: int, M: int)
where reads(rSrc), writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j]
        end
    end
end          

task Inverse(rA: region(ispace(int1d), double),
              m: int)
where reads writes(rA) do
    
    var a_t: &double = [&double](c.malloc(m * m * 8))

    for i = 0, m do
        for j = 0, m do
            a_t[j*m + i] = rA[i*m + j]
        end
    end

    inverse_terra(a_t, m)

    for i = 0, m do
        for j = 0, m do
            rA[j*m + i] = a_t[i*m + j]
        end
    end

    c.free(a_t)
end

task SpMM(rA: region(ispace(int1d), csb_entry),
          rX: region(ispace(int1d), double),
          rY: region(ispace(int1d), double),
          nnz: int, blocksize: int)
where reads(rA, rX), reads writes(rY) do
    for i = 0, nnz do
        for j = 0, blocksize do
            rY[rA[i].rloc*blocksize + j] = rA[i].val * rX[rA[i].cloc*blocksize + j] + rY[rA[i].rloc*blocksize + j]
        end
    end
end

task SpMM_part(rA: region(ispace(int1d), csb_entry),
          rX: region(ispace(int1d), double),
          rY: region(ispace(int1d), double),
          s: int, e: int, blocksize: int)
where reads(rA, rX), reads writes(rY) do
    var xOffset = rX.ispace.bounds.lo/blocksize
    var yOffset = rY.ispace.bounds.lo/blocksize
    for i = s, e do
        for j = 0, blocksize do
            rY[(yOffset + rA[i].rloc) * blocksize + j] = rY[(yOffset + rA[i].rloc) * blocksize + j] + 
                                                         rA[i].val * rX[(xOffset + rA[i].cloc)*blocksize + j]
        end
    end
end
task EigenComp(rA: region(ispace(int1d), double),
               rB: region(ispace(int1d), double),
               rW: region(ispace(int1d), double),
               blocksize: int)
where reads writes(rA, rB, rW) do
    eigencomp_terra(__physical(rA)[0], __fields(rA)[0],
                    __physical(rB)[0], __fields(rB)[0],
                    __physical(rW)[0], __fields(rW)[0],
                    blocksize)
end

task Diag(rSrc: region(ispace(int1d), double),
          rDst: region(ispace(int1d), double),
          m: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, m do
            if i == j then
                rDst[i*m + j] = rSrc[i]
            else
                rDst[i*m + j] = 0.0
            end
        end
    end
end

task Print(rA: region(ispace(int1d), double),
           m: int, n: int)
where reads(rA) do
    for i = 0, m do
        for j = 0, n do
            c.printf("%.4lf", rA[i*n+j])
            if j ~= n-1 then
                c.printf(",")
            end
        end
        c.printf("\n")
    end
end

task Subtract1(rSrc: region(ispace(int1d), double),
               rDst: region(ispace(int1d), double),
               m: int, n: int)
where reads(rSrc), reads writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j] - rDst[i*n + j]
        end
    end
end              

task Subtract1_part(rSrc: region(ispace(int1d), double),
               rDst: region(ispace(int1d), double),
               m: int, n: int, part: int, M: int)
where reads(rSrc), reads writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j] - rDst[i*n + j]
        end
    end
end              

task Subtract2(rSrc: region(ispace(int1d), double),
               rDst: region(ispace(int1d), double),
               m: int, n: int)
where reads(rSrc), reads writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rDst[i*n + j] - rSrc[i*n + j]
        end
    end
end              

task Subtract2_part(rSrc: region(ispace(int1d), double),
                    rDst: region(ispace(int1d), double),
                    m: int, n: int, part: int, M: int)
where reads(rSrc), reads writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[i*n + j] = rDst[i*n + j] - rSrc[i*n + j]
        end
    end
end              

task Multiply(rSrc: region(ispace(int1d), double),
              rDst: region(ispace(int1d), double),
              m: int, n: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j] * rSrc[i*n + j]
        end
    end
end              

task Multiply_part(rSrc: region(ispace(int1d), double),
              rDst: region(ispace(int1d), double),
              m: int, n: int, part: int, M: int)
where reads(rSrc), writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*n + j] * rSrc[i*n + j]
        end
    end
end              

task ReducedNorm(rSrc: region(ispace(int1d), double),
                 rDst: region(ispace(int1d), double),
                 m: int, n: int)
where reads(rSrc), reads writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[j] = rDst[j] + rSrc[i*n + j]
        end
    end

    for i = 0, n do
        rDst[i] = cmath.sqrt(rDst[i])
    end

end

task ReducedNorm_part(rSrc: region(ispace(int1d), double),
                 rDst: region(ispace(int1d), double),
                 m: int, n: int, part: int, M: int)
where reads(rSrc), reduces+(rDst) do
    var s = m*part
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[j] += rSrc[i*n + j]
        end
    end
end

task ReducedNorm_red(rV: region(ispace(int1d), double),
                     n: int)
where reads writes(rV) do
    for i = 0, n do
        rV[i] = cmath.sqrt(rV[i])
    end

end

task UpdateActiveMask(rSrc: region(ispace(int1d), double),
                      rDst: region(ispace(int1d), int),
                      threshold: double, m: int)
where reads(rSrc), reads writes(rDst) do
    var activeN = 0
    for i = 0, m do
        if rSrc[i] > threshold and rDst[i] == 1 then
            rDst[i] = 1
            activeN = activeN + 1
        else
            rDst[i] = 0
        end
    end

    return activeN
end

task GatherActiveVectors(rSrc: region(ispace(int1d), double),
                         rDst: region(ispace(int1d), double),
                         rMsk: region(ispace(int1d), int),
                         m: int, n: int, activeN: int)
where reads(rSrc, rMsk), writes(rDst) do
    var k : int
    for i = 0, m do
        k = 0
        for j = 0, n do
            if rMsk[j] == 1 then
                rDst[i*activeN + k] = rSrc[i*n + j]
                k = k + 1
            end
        end
    end
end

task GatherActiveVectors_part(rSrc: region(ispace(int1d), double),
                              rDst: region(ispace(int1d), double),
                              rMsk: region(ispace(int1d), int),
                              m: int, n: int, activeN: int, part: int, M: int)
where reads(rSrc, rMsk), writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    var k : int
    for i = s, e do
        k = 0
        for j = 0, n do
            if rMsk[j] == 1 then
                rDst[i*activeN + k] = rSrc[i*n + j]
                k = k + 1
            end
        end
    end
end
task ScatterActiveVectors(rSrc: region(ispace(int1d), double),
                          rDst: region(ispace(int1d), double),
                          rMsk: region(ispace(int1d), int),
                          m: int, n: int, activeN: int)
where reads(rSrc, rMsk), writes(rDst) do
    var k : int
    for i = 0, m do
        k = 0
        for j = 0, n do
            if rMsk[j] == 1 then
                rDst[i*n + j] = rSrc[i*activeN + k]
                k = k + 1
            end
        end
    end
end

task ScatterActiveVectors_part(rSrc: region(ispace(int1d), double),
                               rDst: region(ispace(int1d), double),
                               rMsk: region(ispace(int1d), int),
                               m: int, n: int, activeN: int, part: int, M: int)
where reads(rSrc, rMsk), writes(rDst) do
    var s = m*part
    var e = min(s+m, M)
    var k : int
    for i = s, e do
        k = 0
        for j = 0, n do
            if rMsk[j] == 1 then
                rDst[i*n + j] = rSrc[i*activeN + k]
                k = k + 1
            end
        end
    end
end

task CopyBlock(rSrc: region(ispace(int1d), double),
               rDst: region(ispace(int1d), double),
               m: int, n: int, ld: int,
               rOffset: int, cOffset: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[(i+rOffset) * ld + (j+cOffset)] = rSrc[i*n + j]
        end
    end
end               

task MakeSymmetric(rA: region(ispace(int1d), double),
                   m: int)
where reads writes(rA) do
    var avg : double
    for i = 0, m do
        for j = 0, i do
            avg = 0.5*rA[i*m + j] + 0.5*rA[j*m + i]
            rA[i*m + j] = avg
            rA[j*m + i] = avg
        end
    end
end

task CopyCoordX(rSrc: region(ispace(int1d), double),
                rDst: region(ispace(int1d), double),
                m: int, n: int)
where reads(rSrc), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rSrc[i*m + j]
        end
    end
end

task Add(rSrc1: region(ispace(int1d), double),
         rSrc2: region(ispace(int1d), double),
         rDst: region(ispace(int1d), double),
         m: int, n: int)
where reads(rSrc1, rSrc2), writes(rDst) do
    for i = 0, m do
        for j = 0, n do
            rDst[i*n + j] = rSrc1[i*n + j] + rSrc2[i*n + j]
        end
    end
end

task Add_part(rSrc1: region(ispace(int1d), double),
              rSrc2: region(ispace(int1d), double),
              rDst: region(ispace(int1d), double),
              m: int, n: int, part: int, M: int)
where reads(rSrc1, rSrc2), writes(rDst) do
    var s = part*m
    var e = min(s+m, M)
    for i = s, e do
        for j = 0, n do
            rDst[i*n + j] = rSrc1[i*n + j] + rSrc2[i*n + j]
        end
    end
end
task main()

    --[[ simulation parameters ]]--
    var matrix_dim = 1024
    var block_width = 64
    var blocksize = 10
    var maxIterations = 10
    var nnz = 8192
    var alpha_red = 0.0
    var input_file : rawstring
    var args = c.legion_runtime_get_input_args()
    var threshold = 1e-10
    var flag : int
    var nthreads = 14
    
    for i = 0, args.argc do
        if cstr.strcmp(args.argv[i], "-n") == 0 then
            matrix_dim = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-nnz") == 0 then
            nnz = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-b") == 0 then
            block_width = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-blocksize") == 0 then
            blocksize = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-i") == 0 then
            maxIterations = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-graph") == 0 then
            input_file = rawstring(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-nthreads") == 0 then
            nthreads = std.atoi(args.argv[i + 1])
        end
    end
    
    var M = matrix_dim
    var currentBlockSize = blocksize
    var num_subregions = (M + block_width - 1)/block_width
    var gramSize = blocksize + currentBlockSize + currentBlockSize
    var paddedM = num_subregions * block_width

    c.printf("read the command line arguments, paddedM = %d\n", paddedM)

    --[[ region definitions ]]--
    var blk_is = ispace(int1d, num_subregions * num_subregions + 1)
    var mat_is = ispace(int1d, nnz)
    
    var MbyBS_is = ispace(int1d, M * blocksize)
    var MPbyBS_is = ispace(int1d, paddedM * blocksize)
    var MbyCBS_is = ispace(int1d, M * currentBlockSize)
    var MPbyCBS_is = ispace(int1d, paddedM * currentBlockSize)
    var BSbyBS_is = ispace(int1d, blocksize * blocksize)
    var BSbyCBS_is = ispace(int1d, blocksize * currentBlockSize)
    var CBSbyBS_is = ispace(int1d, currentBlockSize * blocksize)
    var CBSbyCBS_is = ispace(int1d, currentBlockSize * currentBlockSize)
    var BSbyMI_is = ispace(int1d, blocksize * maxIterations)
    var BS_is = ispace(int1d, blocksize)
    var GSbyGS_is = ispace(int1d, gramSize * gramSize)
    var GSbyBS_is = ispace(int1d, gramSize * blocksize)
    var GS_is = ispace(int1d, gramSize)

    
    var blkptrs = region(blk_is, int)
    var mat_lr = region(mat_is, csb_entry)
    
    --var blockVectorX = region(MbyBS_is, double)
    var blockVectorX = region(MPbyBS_is, double)
    --var blockVectorAX = region(MbyBS_is, double)
    var blockVectorAX = region(MPbyBS_is, double)
    --var blockVectorR = region(MbyBS_is, double)
    var blockVectorR = region(MPbyBS_is, double)
    --var blockVectorP = region(MbyBS_is, double)
    var blockVectorP = region(MPbyBS_is, double)
    --var blockVectorAP = region(MbyBS_is, double)
    var blockVectorAP = region(MPbyBS_is, double)
    --var newX = region(MbyBS_is, double)
    var newX = region(MPbyBS_is, double)
    --var newAX = region(MbyBS_is, double)
    var newAX = region(MPbyBS_is, double)

    --var activeBlockVectorR = region(MbyCBS_is, double)
    var activeBlockVectorR = region(MPbyCBS_is, double)
    --var activeBlockVectorAR = region(MbyCBS_is, double)
    var activeBlockVectorAR = region(MPbyCBS_is, double)
    --var activeBlockVectorP = region(MbyCBS_is, double)
    var activeBlockVectorP = region(MPbyCBS_is, double)
    --var activeBlockVectorAP = region(MbyCBS_is, double)
    var activeBlockVectorAP = region(MPbyCBS_is, double)
    --var temp3 = region(MbyCBS_is, double)
    var temp3 = region(MPbyCBS_is, double)
    --var newP = region(MbyCBS_is, double)
    var newP = region(MPbyCBS_is, double)
    --var newAP = region(MbyCBS_is, double)
    var newAP = region(MPbyCBS_is, double)

    var gramXAX = region(BSbyBS_is, double)
    var transGramXAX = region(BSbyBS_is, double)
    var gramXBX = region(BSbyBS_is, double)
    var transGramXBX = region(BSbyBS_is, double)
    var lambda = region(BSbyBS_is, double)
    var identity_BB = region(BSbyBS_is, double)

    var gramXAR = region(BSbyCBS_is, double)
    var gramXAP = region(BSbyCBS_is, double)
    var gramXBP = region(BSbyCBS_is, double)
    var zeros_BC = region(BSbyCBS_is, double)
    var temp2 = region(BSbyCBS_is, double)

    var transGramXAR = region(CBSbyBS_is, double)
    var transGramXAP = region(CBSbyBS_is, double)
    var transGramXBP = region(CBSbyBS_is, double)
    var zeros_CB = region(CBSbyBS_is, double)

    var gramRBR = region(CBSbyCBS_is, double)
    var transGramRBR = region(CBSbyCBS_is, double)
    var gramRAR = region(CBSbyCBS_is, double)
    var transGramRAR = region(CBSbyCBS_is, double)
    var gramPBP = region(CBSbyCBS_is, double)
    var transGramPBP = region(CBSbyCBS_is, double)
    var gramRAP = region(CBSbyCBS_is, double)
    var transGramRAP = region(CBSbyCBS_is, double)
    var gramRBP = region(CBSbyCBS_is, double)
    var transGramRBP = region(CBSbyCBS_is, double)
    var gramPAP = region(CBSbyCBS_is, double)
    var identity_PAP = region(CBSbyCBS_is, double)

    var saveLambda = region(BSbyMI_is, double)

    var activeMask = region(BS_is, int)
    var tempLambda = region(BS_is, double)
    var residualNorms = region(BS_is, double)

    var gramA = region(GSbyGS_is, double)
    var transGramA = region(GSbyGS_is, double)
    var gramB = region(GSbyGS_is, double)
    var transGramB = region(GSbyGS_is, double)
    var coordX = region(GSbyBS_is, double)
    var eigenvalue = region(GS_is, double)
    
    --[[ initialization on each region to prevent warning errors ]]--
    fill(blkptrs, 0)
    --[[
    fill(mat_lr.rloc, 0)
    fill(mat_lr.cloc, 0)
    fill(mat_lr.val, 0.0)

    fill(blockVectorX, 0.0)
    fill(blockVectorAX, 0.0)
    fill(blockVectorR, 0.0)
    fill(blockVectorP, 0.0)
    fill(blockVectorAP, 0.0)
    fill(newX, 0.0)
    fill(newAX, 0.0)

    fill(activeBlockVectorR, 0.0)
    fill(activeBlockVectorAR, 0.0)
    fill(activeBlockVectorP, 0.0)
    fill(activeBlockVectorAP, 0.0)
    fill(temp3, 0.0)
    fill(newP, 0.0)
    fill(newAP, 0.0)
    ]]--

    fill(gramXAX, 0.0)
    fill(transGramXAX, 0.0)
    fill(gramXBX, 0.0)
    fill(transGramXBX, 0.0)
    fill(lambda, 0.0)
    fill(identity_BB, 0.0)

    fill(gramXAR, 0.0)
    fill(gramXAP, 0.0)
    fill(gramXBP, 0.0)
    fill(zeros_BC, 0.0)
    fill(temp2, 0.0)

    fill(transGramXAR, 0.0)
    fill(transGramXAP, 0.0)
    fill(transGramXBP, 0.0)
    fill(zeros_CB, 0.0)

    fill(gramRBR, 0.0)
    fill(transGramRBR, 0.0)
    fill(gramRAR, 0.0)
    fill(transGramRAR, 0.0)
    fill(gramPBP, 0.0)
    fill(transGramPBP, 0.0)
    fill(gramRAP, 0.0)
    fill(transGramRAP, 0.0)
    fill(gramRBP, 0.0)
    fill(transGramRBP, 0.0)
    fill(gramPAP, 0.0)
    fill(identity_PAP, 0.0)

    fill(saveLambda, 0.0)
    
    fill(activeMask, 1)
    fill(tempLambda, 0.0)
    fill(residualNorms, 0.0)

    fill(gramA, 0.0)
    fill(transGramA, 0.0)
    fill(gramB, 0.0)
    fill(transGramB, 0.0)
    fill(coordX, 0.0)
    fill(eigenvalue, 0.0)

    --[[ subregion partitioning ]]--
    var ps = ispace(int1d, num_subregions)
    var ts = ispace(int1d, nthreads)

    var mat_ft = partition(equal, mat_lr, ts)
    var blockVectorX_ft = partition(equal, blockVectorX, ts)
    var blockVectorAX_ft = partition(equal, blockVectorAX, ts)
    var blockVectorR_ft = partition(equal, blockVectorR, ts)
    var blockVectorP_ft = partition(equal, blockVectorP, ts)
    var blockVectorAP_ft = partition(equal, blockVectorAP, ts)
    var newX_ft = partition(equal, newX, ts)
    var newAX_ft = partition(equal, newAX, ts)
    var activeBlockVectorR_ft = partition(equal, activeBlockVectorR, ts)
    var activeBlockVectorAR_ft = partition(equal, activeBlockVectorAR, ts)
    var activeBlockVectorP_ft = partition(equal, activeBlockVectorP, ts)
    var activeBlockVectorAP_ft = partition(equal, activeBlockVectorAP, ts)
    var temp3_ft = partition(equal, temp3, ts)
    var newP_ft = partition(equal, newP, ts)
    var newAP_ft = partition(equal, newAP, ts)

    -- initialize vectors in parallel for first touch policy optimization --
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (mat_ft[i]).rloc, 0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (mat_ft[i]).cloc, 0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (mat_ft[i]).val, 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (blockVectorX_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (blockVectorAX_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (blockVectorR_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (blockVectorP_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (blockVectorAP_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (newX_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (newAX_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (activeBlockVectorR_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (activeBlockVectorAR_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (activeBlockVectorP_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (activeBlockVectorAP_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (temp3_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (newP_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (newAP_ft[i]), 0.0)
    end


    var blockVectorX_lp = partition(equal, blockVectorX, ps)
    var blockVectorAX_lp = partition(equal, blockVectorAX, ps)
    var activeBlockVectorR_lp = partition(equal, activeBlockVectorR, ps)
    var activeBlockVectorAR_lp = partition(equal, activeBlockVectorAR, ps)
    var newX_lp = partition(equal, newX, ps)
    var newAX_lp = partition(equal, newAX, ps)
    var blockVectorR_lp = partition(equal, blockVectorR, ps)
    var temp3_lp = partition(equal, temp3, ps)
    var blockVectorP_lp = partition(equal, blockVectorP, ps)
    var activeBlockVectorP_lp = partition(equal, activeBlockVectorP, ps)
    var newP_lp = partition(equal, newP, ps)
    var blockVectorAP_lp = partition(equal, blockVectorAP, ps)
    var activeBlockVectorAP_lp = partition(equal, activeBlockVectorAP, ps)
    var newAP_lp = partition(equal, newAP, ps)
    
    --[[ nonloop parts starts ]]--
    setup_matrix(mat_lr, blkptrs, block_width, num_subregions, input_file)

    GetRandomVectors(blockVectorX, M, blocksize)
    Reset(tempLambda, blocksize, 1)
    Reset(saveLambda, blocksize, maxIterations)
    Identity(identity_BB, blocksize)
    Identity(identity_PAP, currentBlockSize)
    Reset(zeros_BC, blocksize, currentBlockSize)
    Reset(zeros_CB, currentBlockSize, blocksize)

    --XTY(blockVectorX, blockVectorX, gramXBX, blocksize, blocksize, M, 1.0, 0.0)
    for k = 0, num_subregions do
        if k >= 0 then
            XTY_part(blockVectorX_lp[k], blockVectorX_lp[k], gramXBX, blocksize, blocksize, block_width, k, M, 1.0, 0.0)
        end
    end
    Transpose(gramXBX, transGramXBX, blocksize, blocksize)
    Cholesky(transGramXBX, blocksize);
    Transpose(transGramXBX, gramXBX, blocksize, blocksize)
    ResetBelowDiagonal(gramXBX, blocksize)
    Copy(gramXBX, transGramXBX, blocksize, blocksize)
    Inverse(transGramXBX, blocksize)

    --XY(blockVectorX, transGramXBX, newX, M, blocksize, blocksize, 1.0, 0.0)
    for k = 0, num_subregions do
        if k >= 0 then
            XY_part(blockVectorX_lp[k], transGramXBX, newX_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
        end
    end
    --Copy(newX, blockVectorX, M, blocksize)
    __demand(__index_launch)
    for k = 0, num_subregions do
        Copy_part(newX_lp[k], blockVectorX_lp[k], block_width, blocksize, k, M)
    end
    
    --Reset(blockVectorAX, M, blocksize)
    __demand(__index_launch)
    for k = 0, num_subregions do
        Reset_part(blockVectorAX_lp[k], block_width, blocksize, k, M)
    end
    --SpMM(mat_lr, blockVectorX, blockVectorAX, nnz, blocksize)
    for k = 0, num_subregions do
        for j = 0, num_subregions do
            if blkptrs[j * num_subregions + k] < blkptrs[j * num_subregions + k + 1] then
                SpMM_part(mat_lr, blockVectorX_lp[k], blockVectorAX_lp[j], blkptrs[j * num_subregions + k], blkptrs[j * num_subregions + k + 1], blocksize)
            end
        end
    end
    --XTY(blockVectorX, blockVectorAX, gramXAX, blocksize, blocksize, M, 1.0, 0.0)
    for k = 0, num_subregions do
        if k >= 0 then
            XTY_part(blockVectorX_lp[k], blockVectorAX_lp[k], gramXAX, blocksize, blocksize, block_width, k, M, 1.0, 0.0)
        end
    end
    Transpose(gramXAX, transGramXAX, blocksize, blocksize)
    MakeSymmetric(gramXAX, blocksize)

    Transpose(gramXAX, transGramXAX, blocksize, blocksize)
    EigenComp(transGramXAX, identity_BB, tempLambda, blocksize)
    Transpose(transGramXAX, gramXAX, blocksize, blocksize)
    Diag(tempLambda, lambda, blocksize)

    --XY(blockVectorX, gramXAX, newX, M, blocksize, blocksize, 1.0, 0.0)
    for k = 0, num_subregions do
        if k >= 0 then
            XY_part(blockVectorX_lp[k], gramXAX, newX_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
        end
    end
    --Copy(newX, blockVectorX, M, blocksize)
    __demand(__index_launch)
    for k = 0, num_subregions do
        Copy(newX_lp[k], blockVectorX_lp[k], M, blocksize)
    end
    --XY(blockVectorAX, gramXAX, newAX, M, blocksize, blocksize, 1.0, 0.0)
    for k = 0, num_subregions do
        if k >= 0 then
            XY_part(blockVectorAX_lp[k], gramXAX, newAX_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
        end
    end
    --Copy(newAX, blockVectorAX, M, blocksize)
    __demand(__index_launch)
    for k = 0, num_subregions do
        Copy_part(newAX_lp[k], blockVectorAX_lp[k], block_width, blocksize, k, M)
    end
    var timings : double[11] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    __fence(__execution, __block)
    timings[0] = c.legion_get_current_time_in_micros()/1.e6

    for i = 0, maxIterations do
        Reset(residualNorms, blocksize, 1)
        Reset(gramPBP, currentBlockSize, currentBlockSize)
        Reset(gramXAR, blocksize, currentBlockSize)
        Reset(gramRAR, currentBlockSize, currentBlockSize)
        Reset(gramXAP, blocksize, currentBlockSize)
        Reset(gramRAP, currentBlockSize, currentBlockSize)
        Reset(gramPAP, currentBlockSize, currentBlockSize)
        Reset(gramXBP, blocksize, currentBlockSize)
        Reset(gramRBP, currentBlockSize, currentBlockSize)
        --XY(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XY_part(blockVectorX_lp[k], lambda, blockVectorR_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
            end
        end
        --Subtract1(blockVectorAX, blockVectorR, M, blocksize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Subtract1_part(blockVectorAX_lp[k], blockVectorR_lp[k], block_width, blocksize, k, M)
        end
        --Multiply(blockVectorR, newX, M, blocksize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Multiply_part(blockVectorR_lp[k], newX_lp[k], block_width, blocksize, k, M)
        end
        --ReducedNorm(newX, residualNorms, M, blocksize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            ReducedNorm_part(newX_lp[k], residualNorms, block_width, blocksize, k, M)
        end
        ReducedNorm_red(residualNorms, blocksize)
        currentBlockSize = UpdateActiveMask(residualNorms, activeMask, threshold, blocksize)
        
        if currentBlockSize == blocksize then
            flag = 1
        else
            flag = 0
        end
        if i == 0 then
            gramSize = blocksize + currentBlockSize
        else
            gramSize = blocksize + currentBlockSize + currentBlockSize
        end

        if currentBlockSize == 0 then
            c.printf("Converged at iteration #%d\n", i + 1)
            break
        end
        
        --GatherActiveVectors(blockVectorR, activeBlockVectorR, activeMask, M, blocksize, currentBlockSize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            GatherActiveVectors_part(blockVectorR_lp[k], activeBlockVectorR_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
        end
        Reset(temp2, blocksize, currentBlockSize)
        --XTY(blockVectorX, activeBlockVectorR, temp2, blocksize, currentBlockSize, M, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XTY_part(blockVectorX_lp[k], activeBlockVectorR_lp[k], temp2, blocksize, currentBlockSize, block_width, k, M, 1.0, 0.0)
            end
        end
        --XY(blockVectorX, temp2, temp3, M, currentBlockSize, blocksize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XY_part(blockVectorX_lp[k], temp2, temp3_lp[k], block_width, currentBlockSize, blocksize, k, M, 1.0, 0.0)
            end
        end
        --Subtract2(temp3, activeBlockVectorR, M, currentBlockSize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Subtract2_part(temp3_lp[k], activeBlockVectorR_lp[k], block_width, currentBlockSize, k, M)
        end

        --XTY(activeBlockVectorR, activeBlockVectorR, gramRBR, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XTY_part(activeBlockVectorR_lp[k], activeBlockVectorR_lp[k], gramRBR, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
            end
        end
        Transpose(gramRBR, transGramRBR, currentBlockSize, currentBlockSize)
        Cholesky(transGramRBR, currentBlockSize)
        Transpose(transGramRBR, gramRBR, currentBlockSize, currentBlockSize)
        ResetBelowDiagonal(gramRBR, currentBlockSize)
        Inverse(gramRBR, currentBlockSize)
        
        --XY(activeBlockVectorR, gramRBR, temp3, M, currentBlockSize, currentBlockSize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XY_part(activeBlockVectorR_lp[k], gramRBR, temp3_lp[k], block_width, currentBlockSize, currentBlockSize, k, M, 1.0, 0.0)
            end
        end
        --Copy(temp3, activeBlockVectorR, M, currentBlockSize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Copy_part(temp3_lp[k], activeBlockVectorR_lp[k], block_width, currentBlockSize, k, M)
        end
        --ScatterActiveVectors(activeBlockVectorR, blockVectorR, activeMask, M, blocksize, currentBlockSize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            ScatterActiveVectors_part(activeBlockVectorR_lp[k], blockVectorR_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
        end

        --Reset(activeBlockVectorAR, M, currentBlockSize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Reset_part(activeBlockVectorAR_lp[k], block_width, currentBlockSize, k, M)
        end
        --SpMM(mat_lr, activeBlockVectorR, activeBlockVectorAR, nnz, blocksize)
        for k = 0, num_subregions do
            for j = 0, num_subregions do
                if blkptrs[j * num_subregions + k] < blkptrs[j * num_subregions + k + 1] then
                    SpMM_part(mat_lr, activeBlockVectorR_lp[k], activeBlockVectorAR_lp[j], blkptrs[j * num_subregions + k], blkptrs[j * num_subregions + k + 1], blocksize)
                end
            end
        end
        if i ~= 0 then
            --GatherActiveVectors(blockVectorP, activeBlockVectorP, activeMask, M, blocksize, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                GatherActiveVectors_part(blockVectorP_lp[k], activeBlockVectorP_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
            end
            --XTY(activeBlockVectorP, activeBlockVectorP, gramPBP, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(activeBlockVectorP_lp[k], activeBlockVectorP_lp[k], gramPBP, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            Transpose(gramPBP, transGramPBP, currentBlockSize, currentBlockSize)
            Cholesky(transGramPBP, currentBlockSize)
            Transpose(transGramPBP, gramPBP, currentBlockSize, currentBlockSize)
            ResetBelowDiagonal(gramPBP, currentBlockSize)
            Inverse(gramPBP, currentBlockSize)

            --XY(activeBlockVectorP, gramPBP, newP, block_width, currentBlockSize, currentBlockSize, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XY_part(activeBlockVectorP_lp[k], gramPBP, newP_lp[k], block_width, currentBlockSize, currentBlockSize, k, M, 1.0, 0.0)
                end
            end
            --Copy(newP, activeBlockVectorP, M, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                Copy_part(newP_lp[k], activeBlockVectorP_lp[k], block_width, currentBlockSize, k, M)
            end
            --ScatterActiveVectors(activeBlockVectorP, blockVectorP, activeMask, M, blocksize, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                ScatterActiveVectors_part(activeBlockVectorP_lp[k], blockVectorP_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
            end
            
            --GatherActiveVectors(blockVectorAP, activeBlockVectorAP, activeMask, M, blocksize, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                GatherActiveVectors_part(blockVectorAP_lp[k], activeBlockVectorAP_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
            end
            --XY(activeBlockVectorAP, gramPBP, newAP, M, currentBlockSize, currentBlockSize, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XY_part(activeBlockVectorAP_lp[k], gramPBP, newAP_lp[k], block_width, currentBlockSize, currentBlockSize, k, M, 1.0, 0.0)
                end
            end
            --Copy(newAP, activeBlockVectorAP, M, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                Copy_part(newAP_lp[k], activeBlockVectorAP_lp[k], block_width, currentBlockSize, k, M)
            end
            --ScatterActiveVectors(activeBlockVectorAP, blockVectorAP, activeMask, M, blocksize, currentBlockSize)
            __demand(__index_launch)
            for k = 0, num_subregions do
                ScatterActiveVectors_part(activeBlockVectorAP_lp[k], blockVectorAP_lp[k], activeMask, block_width, blocksize, currentBlockSize, k, M)
            end
        end

        --XTY(blockVectorAX, activeBlockVectorR, gramXAR, blocksize, currentBlockSize, M, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XTY_part(blockVectorAX_lp[k], activeBlockVectorR_lp[k], gramXAR, blocksize, currentBlockSize, block_width, k, M, 1.0, 0.0)
            end
        end
        --XTY(activeBlockVectorAR, activeBlockVectorR, gramRAR, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XTY_part(activeBlockVectorAR_lp[k], activeBlockVectorR_lp[k], gramRAR, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
            end
        end
        Transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize)
        MakeSymmetric(gramRAR, currentBlockSize)

        if i == 0 then
            if flag == 1 then
                Transpose(gramXAR, transGramXAR, blocksize, currentBlockSize)
                
                CopyBlock(lambda, gramA, blocksize, blocksize, gramSize, 0, 0)
                CopyBlock(gramXAR, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize)
                CopyBlock(transGramXAR, gramA, currentBlockSize, blocksize, gramSize, blocksize, 0)
                CopyBlock(gramRAR, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize)

                Identity(gramB, gramSize)
            end
        else
            --XTY(blockVectorAX, activeBlockVectorP, gramXAP, blocksize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(blockVectorAX_lp[k], activeBlockVectorP_lp[k], gramXAP, blocksize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            --XTY(activeBlockVectorAR, activeBlockVectorP, gramRAP, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(activeBlockVectorAR_lp[k], activeBlockVectorP_lp[k], gramRAP, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            --XTY(activeBlockVectorAP, activeBlockVectorP, gramPAP, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(activeBlockVectorAP_lp[k], activeBlockVectorP_lp[k], gramPAP, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            MakeSymmetric(gramPAP, currentBlockSize)

            --XTY(blockVectorX, activeBlockVectorP, gramXBP, blocksize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(blockVectorX_lp[k], activeBlockVectorP_lp[k], gramXBP, blocksize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            --XTY(activeBlockVectorR, activeBlockVectorP, gramRBP, currentBlockSize, currentBlockSize, M, 1.0, 0.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XTY_part(activeBlockVectorR_lp[k], activeBlockVectorP_lp[k], gramRBP, currentBlockSize, currentBlockSize, block_width, k, M, 1.0, 0.0)
                end
            end
            
            if flag == 1 then
                Transpose(gramXAR, transGramXAR, blocksize, currentBlockSize)
                Transpose(gramXAP, transGramXAP, blocksize, currentBlockSize)
                Transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize)

                Transpose(gramXBP, transGramXBP, blocksize, currentBlockSize)
                Transpose(gramRBP, transGramRBP, currentBlockSize, currentBlockSize)

                CopyBlock(lambda, gramA, blocksize, blocksize, gramSize, 0, 0)
                CopyBlock(gramXAR, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize)
                CopyBlock(gramXAP, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize+currentBlockSize)
                CopyBlock(transGramXAR, gramA, currentBlockSize, blocksize, gramSize, blocksize, 0)
                CopyBlock(gramRAR, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize)
                CopyBlock(gramRAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize+currentBlockSize)
                CopyBlock(transGramXAP, gramA, currentBlockSize, blocksize, gramSize, blocksize+currentBlockSize, 0)
                CopyBlock(transGramRAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize)
                CopyBlock(gramPAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize+currentBlockSize)

                CopyBlock(identity_BB, gramB, blocksize, blocksize, gramSize, 0, 0)
                CopyBlock(zeros_BC, gramB, blocksize, currentBlockSize, gramSize, 0, blocksize)
                CopyBlock(gramXBP, gramB, blocksize, currentBlockSize, gramSize, 0, blocksize+currentBlockSize)
                CopyBlock(zeros_CB, gramB, currentBlockSize, blocksize, gramSize, blocksize, 0)
                CopyBlock(identity_PAP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize)
                CopyBlock(gramRBP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize+currentBlockSize)
                CopyBlock(transGramXBP, gramB, currentBlockSize, blocksize, gramSize, blocksize+currentBlockSize, 0)
                CopyBlock(transGramRBP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize)
                CopyBlock(identity_PAP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize+currentBlockSize)
            end
        end

        Reset(eigenvalue, gramSize, 1)
        Transpose(gramA, transGramA, gramSize, gramSize)
        Transpose(gramB, transGramB, gramSize, gramSize)
        EigenComp(transGramA, transGramB, eigenvalue, gramSize)
        Transpose(transGramA, gramA, gramSize, gramSize)
        Transpose(transGramB, gramB, gramSize, gramSize)

        Diag(eigenvalue, lambda, blocksize)
        
        CopyCoordX(gramA, coordX, gramSize, blocksize)

        --XYOffset(activeBlockVectorR, coordX, 1, blockVectorP, M, blocksize, currentBlockSize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XYOffset_part(activeBlockVectorR_lp[k], coordX, blocksize*blocksize, blockVectorP_lp[k], block_width, blocksize, currentBlockSize, k, M, 1.0, 0.0)
            end
        end
        --XYOffset(activeBlockVectorAR, coordX, 1, blockVectorAP, M, blocksize, currentBlockSize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XYOffset_part(activeBlockVectorAR_lp[k], coordX, blocksize*blocksize, blockVectorAP_lp[k], block_width, blocksize, currentBlockSize, k, M, 1.0, 0.0)
            end
        end
        if i ~= 0 then
            --XYOffset(activeBlockVectorP, coordX, 2, blockVectorP, M, blocksize, currentBlockSize, 1.0, 1.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XYOffset_part(activeBlockVectorP_lp[k], coordX, (blocksize+currentBlockSize) * blocksize, blockVectorP_lp[k], block_width, blocksize, currentBlockSize, k, M, 1.0, 1.0)
                end
            end
            --XYOffset(activeBlockVectorAP, coordX, 2, blockVectorAP, M, blocksize, currentBlockSize, 1.0, 1.0)
            for k = 0, num_subregions do
                if k >= 0 then
                    XYOffset_part(activeBlockVectorAP_lp[k], coordX, (blocksize+currentBlockSize) * blocksize, blockVectorAP_lp[k], block_width, blocksize, currentBlockSize, k, M, 1.0, 1.0)
                end
            end
        end

        --XY(blockVectorX, coordX, newX, M, blocksize, blocksize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XY_part(blockVectorX_lp[k], coordX, newX_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
            end
        end
        --Add(blockVectorP, newX, blockVectorX, M, blocksize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Add_part(blockVectorP_lp[k], newX_lp[k], blockVectorX_lp[k], block_width, blocksize, k, M)
        end

        --XY(blockVectorAX, coordX, newAX, M, blocksize, blocksize, 1.0, 0.0)
        for k = 0, num_subregions do
            if k >= 0 then
                XY_part(blockVectorAX_lp[k], coordX, newAX_lp[k], block_width, blocksize, blocksize, k, M, 1.0, 0.0)
            end
        end
        --Add(blockVectorAP, newAX, blockVectorAX, M, blocksize)
        __demand(__index_launch)
        for k = 0, num_subregions do
            Add_part(blockVectorAP_lp[k], newAX_lp[k], blockVectorAX_lp[k], block_width, blocksize, k, M)
        end

        CopyBlock(eigenvalue, saveLambda, blocksize, 1, maxIterations, 0, i)

        __fence(__execution, __block)
        timings[i+1] = c.legion_get_current_time_in_micros()/1.e6
    end

    var totalSum = 0.0;
    for i = 0, maxIterations do
        totalSum = totalSum + timings[i+1] - timings[i]
        c.printf("%.4lf,", timings[i+1] - timings[i])
    end
    c.printf("%.4lf\n", totalSum/maxIterations)

    Print(saveLambda, blocksize, maxIterations)
end

regentlib.start(main)
