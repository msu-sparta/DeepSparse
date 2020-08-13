import "regent"

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
    --var irem : &int32 = [&int32](c.malloc(nnonzero * 4))
    var irem : &int32 = [&int32](c.malloc(nnzmalloc))
    --var xrem : &float = [&float](c.malloc(nnonzero * 4))
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
        rP[i + 1] = rP[i] + nnz_block[i];
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

task matvec(rA: region(ispace(int1d), csb_entry),
            rX: region(ispace(int1d), double),
            rY: region(ispace(int1d), double),
            s: int, e: int)
where reads(rA, rX), reads writes (rY) do
    for i = s, e do
        rY[rY.ispace.bounds.lo + rA[i].rloc] = rY[rY.ispace.bounds.lo + rA[i].rloc] 
                                        + rA[i].val * rX[rX.ispace.bounds.lo + rA[i].cloc]
    end
end

task find_norm(rV: region(ispace(int1d), double),
               matrix_dim: int, block_size: int, part: int)
where reads(rV) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    var local_norm = 0.0
    for i = s, e do
         local_norm = local_norm + rV[i] * rV[i]
    end
    return local_norm
end

task normalize(rV: region(ispace(int1d), double), 
               matrix_dim: int, block_size: int,
               part: int, norm: double)
where reads writes(rV) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        rV[i] = rV[i] / norm
    end
end

task copy_vector(rSrc: region(ispace(int1d), double),
                 rDst: region(ispace(int1d), double),
                 matrix_dim: int, block_size: int, part: int)
where reads(rSrc), writes(rDst) do
    var s = part * block_size
    var e = min(s + block_size, matrix_dim)
    for i = s, e do
        rDst[i] = rSrc[i]
    end
end

task main()
    var matrix_dim = 1024
    var block_size = 64
    var num_iterations = 10
    var nnz = 8192
    var norm = 0.0
    var input_file : rawstring
    var args = c.legion_runtime_get_input_args()
    
    for i = 0, args.argc do
        if cstr.strcmp(args.argv[i], "-n") == 0 then
            matrix_dim = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-nnz") == 0 then
            nnz = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-b") == 0 then
            block_size = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-i") == 0 then
            num_iterations = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-graph") == 0 then
            input_file = rawstring(args.argv[i + 1])
        end
    end
    
    var num_subregions = (matrix_dim + block_size - 1)/block_size

    var vec_is = ispace(int1d, num_subregions * block_size)
    var blk_is = ispace(int1d, num_subregions * num_subregions + 1)
    var mat_is = ispace(int1d, nnz)

    var vec_lr = region(vec_is, double)
    var tmp_lr = region(vec_is, double)
    var blkptrs = region(blk_is, int)
    var mat_lr = region(mat_is, csb_entry)

    var ps = ispace(int1d, num_subregions)

    fill(vec_lr, 0.5)
    fill(mat_lr.rloc, 0)
    fill(mat_lr.cloc, 0)
    fill(mat_lr.val, 0.0)

    setup_matrix(mat_lr, blkptrs, block_size, num_subregions, input_file)

    var vec_lp = partition(equal, vec_lr, ps)
    var tmp_lp = partition(equal, tmp_lr, ps)
    
    var norms : double[10] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var timings : double[11] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    __fence(__execution, __block)
    timings[0] = c.legion_get_current_time_in_micros()/1.e6

    c.printf("executing new code\n")
    for i = 0, num_iterations do
        -- Y = 0
        __demand(__index_launch)
        for j = 0, num_subregions do
            fill( (tmp_lp[j]), 0.0)
        end

        -- Y = A*X 
        for k = 0, num_subregions do
            for j = 0, num_subregions do
                if blkptrs[j * num_subregions + k] < blkptrs[j * num_subregions + k + 1] then
                    matvec(mat_lr, vec_lp[k], tmp_lp[j], blkptrs[j * num_subregions + k], blkptrs[j * num_subregions + k + 1])
                end
            end
        end

        -- calculate norm(Y)
        norm = 0.0
        __demand(__index_launch)
        for j = 0, num_subregions do
            norm += find_norm(tmp_lp[j], matrix_dim, block_size, j)
        end
        norms[i] = cmath.sqrt(norm)

        -- Y = Y/norm(Y)
        __demand(__index_launch)
        for j = 0, num_subregions do
            normalize(tmp_lp[j], matrix_dim, block_size, j, norms[i])
        end

        -- X = Y
        __demand(__index_launch)
        for j = 0, num_subregions do
            copy_vector(tmp_lp[j], vec_lp[j], matrix_dim, block_size, j)
        end

        __fence(__execution, __block)
        timings[i+1] = c.legion_get_current_time_in_micros()/1.e6
    end

    var totalSum = 0.0;
    for i = 0, num_iterations do
        totalSum = totalSum + timings[i+1] - timings[i]
        c.printf("%.4lf,", timings[i+1] - timings[i])
    end
    c.printf("%.4lf", totalSum/num_iterations)
    c.printf("\n");
    for i = 0, num_iterations do
        c.printf("%.4lf", norms[i])
        if i < num_iterations - 1 then
            c.printf(",");
        end
    end
    c.printf("\n");
end

regentlib.start(main)
