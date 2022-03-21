import "regent"

local c = regentlib.c
local cstr = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")

local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

task init_matrix(rA: region(ispace(f2d), double))
where writes(rA) do
    for p in rA.ispace do
        rA[p] = p.x * 10.0 + p.y
    end
end

task matvec(rA: region(ispace(f2d), double),
            rX: region(ispace(int1d), double),
            rY: region(ispace(int1d), double))
where reads(rA, rX), reduces +(rY) do
    for p in rA.ispace do
        rY[p.x] += rA[p] * rX[p.y]
    end
end

task copy_vector(rSrc: region(ispace(int1d), double),
                 rDst: region(ispace(int1d), double))
where reads(rSrc), writes(rDst) do
    for i in rSrc.ispace do
        rDst[i] = rSrc[i]
    end
end

task find_norm(rX: region(ispace(int1d), double))
where reads(rX) do
    var local_norm = 0.0
    for i in rX.ispace do
         local_norm = local_norm + rX[i] * rX[i]
    end
    return local_norm
end

task normalize(rX: region(ispace(int1d), double), norm: double)
where reads writes(rX) do
    for i in rX.ispace do
        rX[i] = rX[i] / norm
    end
end

task main()
    var matrix_dim = 1024
    var num_subregions = 16
    var num_iterations = 10
    var norm = 0.0

    var args = c.legion_runtime_get_input_args()
    for i = 0, args.argc do
        if cstr.strcmp(args.argv[i], "-n") == 0 then
            matrix_dim = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-b") == 0 then
            num_subregions = std.atoi(args.argv[i + 1])
        elseif cstr.strcmp(args.argv[i], "-i") == 0 then
            num_iterations = std.atoi(args.argv[i + 1])
        end
    end

    var vec_is = ispace(int1d, matrix_dim)
    var mat_is = ispace(f2d, { x = matrix_dim, y = matrix_dim} )

    var vec_lr = region(vec_is, double)
    var tmp_lr = region(vec_is, double)
    var mat_lr = region(mat_is, double)

    var ps = ispace(int1d, num_subregions)
    var cs = ispace(f2d, { x = num_subregions, y = num_subregions })
    var vec_lp = partition(equal, vec_lr, ps)
    var tmp_lp = partition(equal, tmp_lr, ps)
    var mat_lp = partition(equal, mat_lr, cs)

    -- init X
    __demand(__index_launch)
    for i = 0, num_subregions do 
        fill((vec_lp[i]), 1.0)
    end
    -- init A
    for i = 0, num_subregions do
        for j = 0, num_subregions do
            fill(( mat_lp[ f2d { x = i, y = j } ] ), 0.0)
            init_matrix(mat_lp[ f2d { x = i, y = j } ])
        end
    end
    
    var norms : double[10] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var timings : double[11] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    __fence(__execution, __block)
    timings[0] = c.legion_get_current_time_in_micros()/1.e6

    for i = 0, num_iterations do
        -- Y = 0
        __demand(__index_launch)
        for j = 0, num_subregions do
            fill(( tmp_lp[j] ), 0.0)
        end

        -- Y = A*X 
        -- __demand(__index_launch)
        for j = 0, num_subregions do
            for k = 0, num_subregions do
                matvec(mat_lp[ f2d { x = j, y = k } ], vec_lp[k], tmp_lp[j])
            end
        end

        -- X = Y
        __demand(__index_launch)
        for j = 0, num_subregions do
            copy_vector(tmp_lp[j], vec_lp[j])
        end

        -- calculate norm(X)
        norm = 0.0
        __demand(__index_launch)
        for j = 0, num_subregions do
            norm += find_norm(vec_lp[j])
        end
        norms[i] = cmath.sqrt(norm)

        -- X = X/norm(X)
        __demand(__index_launch)
        for j = 0, num_subregions do
            normalize(vec_lp[j], norms[i])
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
