import "regent"

local blas = terralib.includecstring [[

extern void LAPACKE_dsterf(int n, double *d, double *e);

extern double cblas_ddot(int n, double *x, int incx, double *y, int incy);

extern void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

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
    {rloc, cloc} : uint16,
    val: double,
}

terra get_raw_ptr(index: int, block_size: int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
    var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
    var rect : c.legion_rect_1d_t
    var subrect : c.legion_rect_1d_t
    var offsets : c.legion_byte_offset_t[1]

    rect.lo.x[0] = index * block_size
    rect.hi.x[0] = (index + 1) * block_size - 1

    return [&double](c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, &(offsets[0])))
end

task setup_matrix(rA: region(ispace(int1d), csb_entry),
                  rP: region(ispace(int1d), int),
                  block_size: int32, num_subregions: int32,
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
            blkr = irem[j]/block_size
            blkc = i/block_size
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
            blk = (irem[j]/block_size) * num_subregions + i/block_size
            pos = nnz_block[blk]
            
            rA[pos].rloc = irem[j]%block_size
            rA[pos].cloc = i%block_size
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

task copy_vector(rDst: region(ispace(int1d), double),
                 rSrc: region(ispace(int1d), double),
                 matrix_dim: int, eig_wanted: int, 
                 block_size: int, part: int, offset: int)
where reads(rSrc), writes(rDst) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        rDst[i * eig_wanted + offset] = rSrc[i]
    end
end

task matvec(rA: region(ispace(int1d), csb_entry),
            rX: region(ispace(int1d), double),
            rY: region(ispace(int1d), double),
            s: int, e: int)
where reads(rA, rX), reads writes(rY) do
    for i = s, e do
        rY[rY.ispace.bounds.lo + rA[i].rloc] = rY[rY.ispace.bounds.lo + rA[i].rloc] 
                                        + rA[i].val * rX[rX.ispace.bounds.lo + rA[i].cloc]
    end
end

terra ddot_terra(matrix_dim: int, block_size: int, part: int,
                 prX: c.legion_physical_region_t,
                 fldX: c.legion_field_id_t,
                 prY: c.legion_physical_region_t,
                 fldY: c.legion_field_id_t)
    var s = part * block_size
    var e = min(s+block_size, matrix_dim)
    var ptr_X = get_raw_ptr(part, block_size, prX, fldX)
    var ptr_Y = get_raw_ptr(part, block_size, prY, fldY)

    return blas.cblas_ddot(e - s, ptr_X, 1, ptr_Y, 1)
end

task dot_product(rX: region(ispace(int1d), double),
                 rY: region(ispace(int1d), double),
                 matrix_dim: int, block_size: int, part: int)
where reads(rX, rY) do
    
    --[[
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    var res = 0.0
    for i = s, e do
        res = res + rX[i] * rY[i]
    end
    return res
    ]]--

    return ddot_terra(matrix_dim, block_size, part,
                      __physical(rX)[0], __fields(rX)[0],
                      __physical(rY)[0], __fields(rY)[0]);
end

terra dgemm_transpose_terra(matrix_dim: int, eig_wanted: int,
                            block_size: int, part: int, avail_eigs: int,
                            prA: c.legion_physical_region_t,
                            fldA: c.legion_field_id_t,
                            prX: c.legion_physical_region_t,
                            fldX: c.legion_field_id_t,
                            prY: c.legion_physical_region_t,
                            fldY: c.legion_field_id_t)
    var s = part * block_size
    var e = min(s+block_size, matrix_dim)
    var ptr_A = get_raw_ptr(part, block_size * eig_wanted, prA, fldA)
    var ptr_X = get_raw_ptr(part, block_size, prX, fldX)
    var ptr_Y = get_raw_ptr(0, eig_wanted, prY, fldY)

    blas.cblas_dgemm(101, 112, 111, avail_eigs, 1, e - s,
                1.0, ptr_A, eig_wanted, ptr_X, 1, 1.0, ptr_Y, 1);
end

task custom_dgemm_transpose(rA: region(ispace(int1d), double),
                            rX: region(ispace(int1d), double),
                            rY: region(ispace(int1d), double),
                            matrix_dim: int, eig_wanted: int,
                            block_size: int, part: int, avail_eigs: int)
where reads(rA, rX), reduces+(rY) do
    
    --[[
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        for j = 0, avail_eigs do
            rY[j] += rA[i * eig_wanted + j] * rX[i]
        end
    end
    ]]--

    dgemm_transpose_terra(matrix_dim, eig_wanted, block_size, part, avail_eigs,
                __physical(rA)[0], __fields(rA)[0],
                __physical(rX)[0], __fields(rX)[0],
                __physical(rY)[0], __fields(rY)[0])
end

terra dgemm_terra(matrix_dim: int, eig_wanted: int,
                  block_size: int, part: int, avail_eigs: int,
                  prA: c.legion_physical_region_t,
                  fldA: c.legion_field_id_t,
                  prX: c.legion_physical_region_t,
                  fldX: c.legion_field_id_t,
                  prY: c.legion_physical_region_t,
                  fldY: c.legion_field_id_t)
    var s = part * block_size
    var e = min(s+block_size, matrix_dim)
    var ptr_A = get_raw_ptr(part, block_size * eig_wanted, prA, fldA)
    var ptr_X = get_raw_ptr(0, eig_wanted, prX, fldX)
    var ptr_Y = get_raw_ptr(part, block_size, prY, fldY)

    blas.cblas_dgemm(101, 111, 111, e - s, 1, avail_eigs,
                1.0, ptr_A, eig_wanted, ptr_X, 1, 0.0, ptr_Y, 1)
end

task custom_dgemm(rA: region(ispace(int1d), double),
                  rX: region(ispace(int1d), double),
                  rY: region(ispace(int1d), double),
                  matrix_dim: int, eig_wanted: int,
                  block_size: int, part: int, avail_eigs: int)
where reads(rA, rX), reads writes(rY) do
    
    --[[
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        for j = 0, avail_eigs do
            rY[i] = rY[i] + rA[i * eig_wanted + j] * rX[j]
        end
    end
    ]]--
    
    dgemm_terra(matrix_dim, eig_wanted, block_size, part, avail_eigs,
                __physical(rA)[0], __fields(rA)[0],
                __physical(rX)[0], __fields(rX)[0],
                __physical(rY)[0], __fields(rY)[0])
end

task subtract_vector(rX: region(ispace(int1d), double),
                     rY: region(ispace(int1d), double),
                     matrix_dim: int, block_size: int, part: int)
where reads writes(rX), reads(rY) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        rX[i] = rX[i] - rY[i]
    end
end

task find_norm(rV: region(ispace(int1d), double),
               matrix_dim: int, block_size: int, part: int)
where reads(rV) do
    
    --[[
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    var local_norm = 0.0
    for i = s, e do
         local_norm = local_norm + rV[i] * rV[i]
    end
    return local_norm
    ]]--
    
    return ddot_terra(matrix_dim, block_size, part,
                      __physical(rV)[0], __fields(rV)[0],
                      __physical(rV)[0], __fields(rV)[0]);
end

task normalize(rX: region(ispace(int1d), double), 
               rY: region(ispace(int1d), double), 
               matrix_dim: int, block_size: int,
               part: int, norm: double)
where writes(rX), reads(rY) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        rX[i] = rY[i] / norm
    end
end

terra dsterf_terra(eig_wanted:int, alpha: &double, beta: &double)
    var eig_wanted_ : int[1]
    var info : int[1]

    eig_wanted_[0] = eig_wanted
    blas.LAPACKE_dsterf(eig_wanted, alpha, beta);
end

task main()
    var matrix_dim = 1024
    var block_size = 64
    var eig_wanted = 10
    var nthreads = 14
    var nnz = 8192
    var norm = 0.0
    var alpha_red = 0.0
    var input_file : rawstring
    var args = c.legion_runtime_get_input_args()
    
    for i = 0, args.argc do
        if cstr.strcmp(args.argv[i], "-n") == 0 then
            matrix_dim = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-nnz") == 0 then
            nnz = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-b") == 0 then
            block_size = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-eig_wanted") == 0 then
            eig_wanted = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-nthreads") == 0 then
            nthreads = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-graph") == 0 then
            input_file = rawstring(args.argv[i + 1])
        end
    end
    
    var num_subregions = (matrix_dim + block_size - 1)/block_size

    c.printf("read the command line argument\n")

    var QpZ_is = ispace(int1d, eig_wanted)
    var vec_is = ispace(int1d, num_subregions * block_size)
    var Q_is = ispace(int1d, num_subregions * block_size * eig_wanted)
    
    var blk_is = ispace(int1d, num_subregions * num_subregions + 1)
    var mat_is = ispace(int1d, nnz)

    var q_lr = region(vec_is, double)
    var qq_lr = region(vec_is, double)
    var z_lr = region(vec_is, double)
    var QQpZ_lr = region(vec_is, double)
    var QpZ_lr = region(QpZ_is, double)
    var Q_lr = region(Q_is, double)

    var blkptrs = region(blk_is, int)
    var mat_lr = region(mat_is, csb_entry)

    c.printf("defined all logical regions\n")

    var ps = ispace(int1d, num_subregions)
    var ts = ispace(int1d, nthreads)

    var qq_lp = partition(equal, qq_lr, ps)
    var z_lp = partition(equal, z_lr, ps)
    var QQpZ_lp = partition(equal, QQpZ_lr, ps)
    var Q_lp = partition(equal, Q_lr, ps)

    var mat_ft = partition(equal, mat_lr, ts)
    var q_ft = partition(equal, q_lr, ts)
    var qq_ft = partition(equal, qq_lr, ts)
    var z_ft = partition(equal, z_lr, ts)
    var Q_ft = partition(equal, Q_lr, ts)
    var QQpZ_ft = partition(equal, QQpZ_lr, ts)

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
        fill( (q_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (qq_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (z_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (Q_ft[i]), 0.0)
    end
    __demand(__index_launch)
    for i = 0, nthreads do
        fill( (QQpZ_ft[i]), 0.0)
    end
    
    setup_matrix(mat_lr, blkptrs, block_size, num_subregions, input_file)

    c.printf("partitioning completed\n")
    
    -- q = ones(numcols)
    fill(q_lr, 1)

    -- qq = q / ||q||
    fill(qq_lr, 1.0/cmath.sqrt(matrix_dim))

    -- Q[column 0] = qq
    for i = 0, num_subregions do
        -- edit
        copy_vector(Q_lp[i], qq_lp[i], matrix_dim, eig_wanted, block_size, i, 0)
    end

    var alpha : double[20] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var beta : double[20] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var timings : double[21] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    c.printf("matrix_order = %d nnz =  %d eig_wanted = %d block_size = %d\n", matrix_dim, nnz, eig_wanted, block_size)
    __fence(__execution, __block)
    timings[0] = c.legion_get_current_time_in_micros()/1.e6

    for i = 0, eig_wanted do

        fill(QpZ_lr, 0.0)
    
        -- z = 0
        __demand(__index_launch)
        for j = 0, num_subregions do
            fill( (z_lp[j]), 0.0)
        end

        -- z = A * qq 
        -- __demand(__index_launch)
        for k = 0, num_subregions do
            for j = 0, num_subregions do
                if blkptrs[j * num_subregions + k] < blkptrs[j * num_subregions + k + 1] then
                    matvec(mat_lr, qq_lp[k], z_lp[j], blkptrs[j * num_subregions + k], blkptrs[j * num_subregions + k + 1])
                end
            end
        end

        -- alpha[iter] = qq' * z
        alpha_red = 0.0
        __demand(__index_launch)
        for j = 0, num_subregions do
           alpha_red += dot_product(qq_lp[j], z_lp[j], matrix_dim, block_size, j)
        end
        alpha[i] = alpha_red;

        -- QpZ = Q' * z
        __demand(__index_launch)
        for j = 0, num_subregions do
            custom_dgemm_transpose(Q_lp[j], z_lp[j], QpZ_lr, matrix_dim, eig_wanted, block_size, j, i + 1)
        end

        -- QQpZ = 0
        __demand(__index_launch)
        for j = 0, num_subregions do
            fill( (QQpZ_lp[j]), 0.0)
        end

        -- QQpZ = Q * QpZ
        __demand(__index_launch)
        for j = 0, num_subregions do
            custom_dgemm(Q_lp[j], QpZ_lr, QQpZ_lp[j], matrix_dim, eig_wanted, block_size, j, i + 1)
        end

        -- z = z - QQpZ
        __demand(__index_launch)
        for j = 0, num_subregions do
            subtract_vector(z_lp[j], QQpZ_lp[j], matrix_dim, block_size, j);
        end

        -- beta[iter] = ||z||
        norm = 0.0
        __demand(__index_launch)
        for j = 0, num_subregions do
            norm += find_norm(z_lp[j], matrix_dim, block_size, j)
        end
        beta[i] = cmath.sqrt(norm)

        -- qq = z / beta[iter]
        __demand(__index_launch)
        for j = 0, num_subregions do
            normalize(qq_lp[j], z_lp[j], matrix_dim, block_size, j, beta[i])
        end

        -- Q[column iter + 1] = qq
        for j = 0, num_subregions do
            copy_vector(Q_lp[j], qq_lp[j], matrix_dim, eig_wanted, block_size, j, i + 1)
        end

        __fence(__execution, __block)
        timings[i+1] = c.legion_get_current_time_in_micros()/1.e6
    end

    
    var totalSum = 0.0;
    for i = 0, eig_wanted do
        totalSum = totalSum + timings[i+1] - timings[i]
        c.printf("%.4lf,", timings[i+1] - timings[i])
    end
    c.printf("%.4lf\n", totalSum/eig_wanted)
    
    for i = 0, eig_wanted do
        c.printf("%.4lf", alpha[i])
        if i < eig_wanted - 1 then
            c.printf(",");
        end
    end
    c.printf("\n");
    
    for i = 0, eig_wanted do
        c.printf("%.4lf", beta[i])
        if i < eig_wanted - 1 then
            c.printf(",");
        end
    end
    c.printf("\n");

    dsterf_terra(eig_wanted, alpha, beta)

    for i = 0, eig_wanted do
        c.printf("%.4lf", alpha[i])
        if i < eig_wanted - 1 then
            c.printf(",");
        end
    end
    c.printf("\n");

end

regentlib.start(main)
