#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/serialization.hpp>
#include <hpx/modules/type_support.hpp>

#include <boost/shared_array.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

bool print_eigs = false;
char const* gather_basename = "/dist_power_iteration/gather/";
char const* reduce_basename = "/dist_power_iteration/reduce/";
char const* all_gather_basename = "/dist_power_iteration/all_gather/";


///////////////////////////////////////////////////////////////////////////////
// a class that holds data for each vector block
struct block_data
{
private:
    typedef hpx::serialization::serialize_buffer<double> buffer_type;

public:
    block_data()
      : size_(0)
    {}

    // Create a new (uninitialized) block of the given size.
    explicit block_data(std::size_t size)
      : data_(std::allocator<double>().allocate(size), size, buffer_type::take),
        size_(size)
    {}

    // Create a new (initialized) block of the given size.
    block_data(std::size_t size, double initial_value)
      : data_(std::allocator<double>().allocate(size), size, buffer_type::take),
        size_(size)
    {
        for (std::size_t i = 0; i != size; ++i)
            data_[i] = initial_value;
    }

    double& operator[](std::size_t idx) { return data_[idx]; }
    double operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return size_; }

private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & data_ & size_;
        // clang-format on
    }

private:
    buffer_type data_;
    std::size_t size_;
};


///////////////////////////////////////////////////////////////////////////////
struct matrix_block
{
private:
    typedef hpx::serialization::serialize_buffer<unsigned short int> loc_buffer_type;
    //typedef hpx::serialization::serialize_buffer<int> loc_buffer_type;
    typedef hpx::serialization::serialize_buffer<double> val_buffer_type;

public:
    matrix_block()
        : nnz_(0)
    {}

    matrix_block(std::size_t nnz)
        : nnz_(nnz),
        rloc_(std::allocator<unsigned short int>().allocate(nnz), nnz, loc_buffer_type::take),
        cloc_(std::allocator<unsigned short int>().allocate(nnz), nnz, loc_buffer_type::take),
        //rloc_(std::allocator<int>().allocate(nnz), nnz, loc_buffer_type::take),
        //cloc_(std::allocator<int>().allocate(nnz), nnz, loc_buffer_type::take),
        val_(std::allocator<double>().allocate(nnz), nnz, val_buffer_type::take)
    {}
    
private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & nnz_ & rloc_ & cloc_ & val_;
        // clang-format on
    }

public:
    std::size_t nnz_;
    loc_buffer_type rloc_;
    loc_buffer_type cloc_;
    val_buffer_type val_;
};


///////////////////////////////////////////////////////////////////////////////
// This is the server side representation of the data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
struct block_server
  : hpx::components::component_base<block_server>
{
    // construct new instances
    block_server() = default;

    explicit block_server(block_data const& data)
      : data_(data)
    {}

    block_server(std::size_t size, double initial_value)
      : data_(size, initial_value)
    {}

    block_data get_data() const
    {
        return data_;
    }

    // Every member function which has to be invoked remotely needs to be
    // wrapped into a component action. The macro below defines a new type
    // 'get_data_action' which represents the (possibly remote) member function
    // block::get_data().
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(block_server, get_data, get_data_action);

private:
    block_data data_;
};

// The macros below are necessary to generate the code required for exposing
// our block type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<block_server> block_server_type;
HPX_REGISTER_COMPONENT(block_server_type, block_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef block_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);


///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate while referencing a remote block.
struct block : hpx::components::client_base<block, block_server>
{
    typedef hpx::components::client_base<block, block_server> base_type;

    block() = default;

    // Create new component on locality 'where' and initialize the held data
    block(hpx::id_type where, std::size_t size, double initial_value)
      : base_type(hpx::new_<block_server>(where, size, initial_value))
    {}

    // Create a new component on the locality co-located to the id 'where'. The
    // new instance will be initialized from the given block_data.
    block(hpx::id_type where, block_data const& data)
      : base_type(hpx::new_<block_server>(hpx::colocated(where), data))
    {}

    // Attach a future representing a (possibly remote) block.
    block(hpx::future<hpx::id_type> && id) noexcept
      : base_type(std::move(id))
    {}

    // Unwrap a future<block> (a block already is a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    block(hpx::future<block> && c) noexcept
      : base_type(std::move(c))
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<block_data> get_data() const
    {
        block_server::get_data_action act;
        return hpx::async(act, get_id());
    }
};


///////////////////////////////////////////////////////////////////////////////
// Data for each iteration on one locality
struct iterate_server : hpx::components::component_base<iterate_server>
{
    // Our data for each vector
    typedef std::vector<block> Vec;
    typedef std::vector<matrix_block> Mat;

    // Construct new instances
    iterate_server() = default;

    explicit iterate_server(std::size_t nl)
    {}

    // Do all the work on 'np' blocks, 'nx' data points each, for 'nt'
    // iterations, limit depth of dependency tree to 'nd'.
    Vec do_work(std::size_t nl, std::size_t block_size, std::size_t nt, 
            std::size_t buffer, std::string str_file);

    HPX_DEFINE_COMPONENT_ACTION(iterate_server, do_work, do_work_action);

protected:
    static void read_custom(int &numrows, int &numcols, int &nnonzero,
            char* filename, int *&colptrs, int *&irem, double *&xrem, 
            int nl, int block_size);
    static void csc2blkcoord(iterate_server::Mat &matrixBlock,
            int *colptrs, int *irem, double *xrem, int numcols,
            int base_rowblk, int nrowblks, int ncolblks, int block_size);
    static block reset_block(block const& V);
    static block copy_block(block const& src, block const& dst);
    static block accumulate_block(block const& src, block const& dst);
    static block matvec_block(matrix_block const& A, block const& X, block const& Y);
    static double dot_block(block const& V);
    static block normalize_block(block const& V, hpx::shared_future<double> norm_ftr);

private:
    Vec X, Y, Y_buffer;
    Mat A;
};

typedef hpx::components::component<iterate_server> iterate_server_type;
HPX_REGISTER_COMPONENT(iterate_server_type, iterate_server);

typedef iterate_server::do_work_action do_work_action;
HPX_REGISTER_ACTION(do_work_action);


///////////////////////////////////////////////////////////////////////////////
// This is a client side member function can now be implemented as the
// iterate_server has been defined.
struct iterate : hpx::components::client_base<iterate, iterate_server>
{
    typedef hpx::components::client_base<iterate, iterate_server> base_type;

    // construct new instances/wrap existing iterates from other localities
    explicit iterate(std::size_t num_localities)
      : base_type(hpx::new_<iterate_server>(hpx::find_here(), num_localities))
    {
    }

    iterate(hpx::future<hpx::id_type> && id) noexcept
      : base_type(std::move(id))
    {}

    ~iterate()
    {
    }

    hpx::future<iterate_server::Vec> do_work(std::size_t nl,
            std::size_t block_size, std::size_t nt, 
            std::size_t buffer, std::string str_file)
    {
        return hpx::async(do_work_action(), get_id(), nl, block_size, nt, 
                buffer, str_file);
    }
};


///////////////////////////////////////////////////////////////////////////////
void iterate_server::read_custom(int &numrows, int &numcols, int &nnonzero, 
        char* filename, int *&colptrs, int *&irem, double *&xrem, 
        int nl, int block_size)
{
    FILE *fp, *fq;
    int *colptrs_global;
    double *xrem_local;
    std::size_t locality_id = hpx::get_locality_id();
    /* 
     * We have to use a buffer to read the irem and xrem of the matrix 
     * in CSC format as reading an entire array causes OOM given that 
     * there are many threads in the some node doing that and 
     * reading the elements of array one by one is just too slow, especially with
     * fread as fread is not designed for that 
     *
     * buffer_size corresponds to the number of columns that we will be reading the
     * nonzeros of at once
    */
    const int buffer_size = 50000;  

    fp = fopen(filename, "rb");
    fq = fopen(filename, "rb");

    if (fp == NULL)
    {
        std::cout << "invalid matrix file name" << std::endl;
        return;
    }

    fread(&numrows, sizeof(int), 1, fp);
    fseek(fq, sizeof(int), SEEK_CUR);
    fread(&numcols, sizeof(int), 1, fp);
    fseek(fq, sizeof(int), SEEK_CUR);
    fread(&nnonzero, sizeof(int), 1, fp);
    fseek(fq, sizeof(int), SEEK_CUR);

    if (numrows != numcols)
    {
        std::cout << "matrix should be square" << std::endl;
        return;
    }

    if (!locality_id)
    {
        std::cout << "row: " << numrows << std::endl;
        std::cout << "column: "<< numcols << std::endl;
        std::cout << "non zero: " << nnonzero << std::endl;
    }

    int dim = numcols;
    std::size_t np = (dim + block_size - 1)/block_size;
    std::size_t local_np = np/nl + (locality_id < np % nl);
    std::size_t base_rowblk = ((np/nl)*locality_id + std::min(locality_id, np%nl));

    colptrs = new int[numcols + 1];
    colptrs_global = new int[numcols + 1];
    
    fread(colptrs_global, sizeof(int), numcols+1, fp);
    fseek(fq, sizeof(int) * (numcols+1), SEEK_CUR);

    for(int i = 0 ; i != numcols+1; i++)
    {   
        colptrs_global[i]--;
    }

    int blkr, nnonzero_local;
    int next_row, start, end;
    float next_nnz;
    int *next_rows;
    float *next_nnzs;

    nnonzero_local = 0;
    next_rows = NULL;
    for(int i = 0 ; i < numcols; i++)
    {
        /* single fread is too slow, read columns 50000 by 50000 */
        if(i%buffer_size == 0)
        {
            if(next_rows != NULL)
            {
                delete []next_rows;
            }
            start = colptrs_global[i];
            end = colptrs_global[std::min(i+buffer_size, numcols)];
            next_rows = new int[end-start];
            fread(next_rows, sizeof(int), end-start, fp);
        }
        colptrs[i] = nnonzero_local;
        for(int j = colptrs_global[i]; j < colptrs_global[i+1]; ++j)
        {
            next_row = next_rows[j-start] - 1;
            blkr = next_row/block_size - base_rowblk;
            if(blkr >= 0 && blkr < local_np)
            {
                nnonzero_local++;
            }
        }
    }
    colptrs[numcols] = nnonzero_local;
    if(next_rows != NULL)
    {
        delete []next_rows;
    }

    irem = new int[nnonzero_local];
    xrem = new double[nnonzero_local];

    fseek(fp, -1 * sizeof(int) * nnonzero, SEEK_CUR);
    fseek(fq, sizeof(int) * nnonzero, SEEK_CUR);

    nnonzero_local = 0;
    next_rows = NULL;
    next_nnzs = NULL;
    for(int i = 0; i < numcols; ++i)
    {
        /* single fread is too slow, read columns 50000 by 50000 */
        if(i%buffer_size == 0)
        {
            if(next_rows != NULL)
            {
                delete []next_rows;
                delete []next_nnzs;
            }
            start = colptrs_global[i];
            end = colptrs_global[std::min(i+buffer_size, numcols)];
            next_rows = new int[end-start];
            next_nnzs = new float[end-start];
            fread(next_rows, sizeof(int), end-start, fp);
            fread(next_nnzs, sizeof(float), end-start, fq);
        }
        for(int j = colptrs_global[i]; j < colptrs_global[i+1]; ++j)
        {
            next_row = next_rows[j-start] - 1;
            next_nnz = next_nnzs[j-start];
            blkr = next_row/block_size - base_rowblk;
            if(blkr >= 0 && blkr < local_np)
            {
                irem[nnonzero_local] = next_row;
                xrem[nnonzero_local] = next_nnz;
                nnonzero_local++;
            }
        }
    }
    if(next_rows != NULL)
    {
        delete []next_rows;
        delete []next_nnzs;
    }

    nnonzero = nnonzero_local;

    fclose(fp);
    fclose(fq);
    delete []colptrs_global;
}

///////////////////////////////////////////////////////////////////////////////
void iterate_server::csc2blkcoord(iterate_server::Mat &matrixBlock,
        int *colptrs, int *irem, double *xrem, int numcols,
        int base_rowblk, int nrowblks, int ncolblks, int block_size)
{
    // std::cout << numcols << " cols, " << base_rowblk << " base row blocks, " 
    //     << nrowblks << " row blocks, " << ncolblks << " column blocks, "
    //     << block_size << " block size" << std::endl;
    int i, j, r, c, k, k1, k2, blkr, blkc;

    int **top;
    top = new int*[nrowblks];
    for (i = 0; i != nrowblks; i++)
    {
        top[i] = new int[ncolblks];
    }

    for (blkr = 0; blkr != nrowblks; blkr++)
    {
        for (blkc = 0 ; blkc != ncolblks ; blkc++)
        {
            top[blkr][blkc] = 0;
        }
    }
    
    /* calculatig nnz per block */
    for (c = 0; c != numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1];
        blkc = c / block_size;

        for (k = k1; k != k2 ; k++)
        {
            r = irem[k];
            blkr = r / block_size - base_rowblk;
            if(blkr >= 0 && blkr < nrowblks)
            {
                top[blkr][blkc]++;  
            }    
        }
    }

    // std::cout << "finished counting nnz in each block" << std::endl << std::flush;

    for (blkc = 0; blkc != ncolblks; blkc++)
    {
        for (blkr = 0; blkr != nrowblks; blkr++)
        {
            if (top[blkr][blkc] != 0)
            {
                matrixBlock[blkr * ncolblks + blkc] = matrix_block(top[blkr][blkc]);
                top[blkr][blkc] = 0;
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc] = matrix_block();
            }
        }
    }

    // std::cout << "allocated memory for each block" << std::endl << std::flush;

    for (c = 0; c != numcols; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1]; 
        blkc = c/block_size;

        for (k = k1; k != k2 ; k++)
        {
            r = irem[k];
            blkr = r / block_size - base_rowblk;
            if(blkr >= 0 && blkr < nrowblks)
            {
	            matrixBlock[blkr * ncolblks + blkc].rloc_[top[blkr][blkc]] = r - (base_rowblk + blkr) * block_size;
                matrixBlock[blkr * ncolblks + blkc].cloc_[top[blkr][blkc]] = c - blkc * block_size;
                matrixBlock[blkr * ncolblks + blkc].val_[top[blkr][blkc]] = xrem[k];
                top[blkr][blkc]++;
            }
        }
    }

    // printf("conversion completed\n\n");

    for(i = 0 ; i != nrowblks; i++)
    {
        delete [] top[i];
    }
    delete [] top;
}


///////////////////////////////////////////////////////////////////////////////
block iterate_server::reset_block(block const& V)
{
    hpx::shared_future<block_data> V_data = V.get_data();

    return V_data.then(
        hpx::util::unwrapping(
            [V](block_data const& V_avail) -> block
            {
                std::size_t size = V_avail.size();
                block_data next(size);
                for (std::size_t i = 0; i != size; ++i)
                {
                    next[i] = 0.0;
                }

                return block(V.get_id(), next);
            }
        )
   );
}


///////////////////////////////////////////////////////////////////////////////
block iterate_server::copy_block(block const& src, block const& dst)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [src, dst](block_data const& src_avail, block_data dst_avail) -> block
            {
                HPX_UNUSED(src);       

                std::size_t size = dst_avail.size();
                for (std::size_t i = 0; i != size; ++i)
                {
                    dst_avail[i] = src_avail[i];
                }

                return block(dst.get_id(), dst_avail);
            }
        ),
        src.get_data(),
        dst.get_data()
   );
}


///////////////////////////////////////////////////////////////////////////////
block iterate_server::accumulate_block(block const& src, block const& dst)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [src, dst](block_data const& src_avail, block_data dst_avail) -> block
            {
                HPX_UNUSED(src);       

                std::size_t size = dst_avail.size();
                for (std::size_t i = 0; i != size; ++i)
                {
                    dst_avail[i] += src_avail[i];
                }

                return block(dst.get_id(), dst_avail);
            }
        ),
        src.get_data(),
        dst.get_data()
   );
}

///////////////////////////////////////////////////////////////////////////////
block iterate_server::matvec_block(matrix_block const& A, block const& X, 
        block const& Y)
{
    
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [A, X, Y](block_data const& X_avail, block_data const& Y_avail) -> block
            {
                HPX_UNUSED(X);

                block_data next(Y_avail);
                for (std::size_t i = 0; i != A.nnz_; ++i)
                {
                    next[A.rloc_[i]] += A.val_[i] * X_avail[A.cloc_[i]];
                }

                return block(Y.get_id(), next);
            }
        ),
        X.get_data(),
        Y.get_data()
   );
}

///////////////////////////////////////////////////////////////////////////////
double iterate_server::dot_block(block const& V)
{
    hpx::shared_future<block_data> V_data = V.get_data();

    return V_data.then(
        hpx::util::unwrapping(
            [V](block_data const& V_avail) -> double
            {
                double rv = 0.0;
                std::size_t size = V_avail.size();
                for (std::size_t i = 0; i != size; ++i)
                {
                    rv += V_avail[i] * V_avail[i];
                }

                return rv;
            }
        )
   ).get();
}


///////////////////////////////////////////////////////////////////////////////
block iterate_server::normalize_block(block const& V, hpx::shared_future<double> norm_ftr)
{
    double norm = sqrt(norm_ftr.get());
    hpx::shared_future<block_data> V_data = V.get_data();

    return V_data.then(
        hpx::util::unwrapping(
            [V, norm](block_data const& V_avail) -> block
            {
                std::size_t size = V_avail.size();
                block_data next(size);
                for (std::size_t i = 0; i != size; ++i)
                {
                    next[i] = V_avail[i]/norm;
                }

                return block(V.get_id(), next);
            }
        )
   );
}


///////////////////////////////////////////////////////////////////////////////
// This is the implementation of main loop
//
// Do the work on 'np/nl' blocks, 'nx' data points each, for 'nt' iterations.
iterate_server::Vec iterate_server::do_work(std::size_t nl,
        std::size_t block_size, std::size_t nt, std::size_t buffer,
        std::string str_file)
{
    // read sparse matrix in csc format
    int numrows, numcols, nnonzero;
    int len = str_file.length();
    char *file_name = new char[len + 1];
    strcpy(file_name, str_file.c_str());
    int *colptrs, *irem;
    double *xrem;
    read_custom(numrows, numcols, nnonzero, file_name, colptrs, irem, xrem, nl, block_size);

    int dim = numcols;

    // initialize vector and matrix block counts, etc.
    std::size_t locality_id = hpx::get_locality_id();
    std::size_t np = (dim + block_size - 1)/block_size;
    std::size_t local_np = np/nl + (locality_id < np % nl);
    std::size_t last_block = (dim%block_size) ? dim%block_size : block_size;
    std::size_t base_rowblk = ((np/nl)*locality_id + std::min(locality_id, np%nl));
    std::uint64_t const num_worker_threads = hpx::get_num_worker_threads();
    //std::size_t num_threads = np;
    std::size_t num_threads = num_worker_threads/nl;
    num_threads *= buffer;
    num_threads = std::min(num_threads, local_np*np);
    
    // allocate space for the vectors and the matrix
    X.resize(local_np);
    Y.resize(local_np);
    Y_buffer.resize(num_threads);
    A.resize(local_np * np);
    
    // convert the matrix to csb format
    csc2blkcoord(A, colptrs, irem, xrem, dim, base_rowblk, local_np, np, block_size);

    int *map_locality = new int[np];
    int *map_block = new int[np];
    std::size_t cnt = 0;

    // map the global rank of a vector block to the
    // locality id it resides as well as its local
    // rank in that locality (used to process all_gather)
    for (std::size_t i = 0; i < nl; ++i)
    {
        for (std::size_t j = 0; j < np/nl + (i < np % nl); ++j)
        {
            map_locality[cnt] = i;
            map_block[cnt] = j;
            ++cnt;
        }
    }

    // Initial values: X[i] = 1.0
    hpx::id_type here = hpx::find_here();
    for (std::size_t i = 0; i != local_np; ++i)
    {
        X[i] = block(here, block_size, 1.0);
    }
    for (std::size_t i = 0; i != num_threads; ++i)
    {
        Y_buffer[i] = block(here, block_size, 0.0);
    }
    
    hpx::wait_all(X);
    hpx::wait_all(Y_buffer);
    
    std::vector<hpx::future<double>> sq_partial_norm_ftr(local_np);
    hpx::future<double> sq_local_norm_ftr;
    hpx::shared_future<double> sq_global_norm_ftr;
    std::vector<hpx::shared_future<double>> norms (nt);
    hpx::future<std::vector<iterate_server::Vec>> global_X_ftr;
    std::vector<iterate_server::Vec> global_X;

    hpx::chrono::high_resolution_timer timer;

    for (std::size_t t = 0; t != nt; ++t)
    {
        for (std::size_t i = 0; i != local_np; ++i)
        {
            // reset ith block of Y when ith block of X is ready
            // to set the dependency between subsequent iterations
            Y[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::reset_block,
                    X[i]
                );
        }

        // all_gather X partitions/blocks from all localities
        // hpx::future<std::vector<iterate_server::Vec> > global_X =
        global_X_ftr = hpx::lcos::all_gather(all_gather_basename, X, nl, t);
     
        /* do local SpMV computations using local X blocks */
        for (std::size_t i = 0; i != local_np; ++i)
        {
            for (std::size_t j = 0; j != local_np; ++j)
            {
                if(A[i*np + j + base_rowblk].nnz_ > 0)
                {
                    std::size_t buffer_target = (i*np + j + base_rowblk) % num_threads;
                    Y_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::matvec_block,
                            A[i*np + j + base_rowblk],
                            X[j],
                            Y_buffer[buffer_target]
                        );
                    Y[i] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::accumulate_block,
                            Y[i], Y_buffer[buffer_target]
                        );
                    Y_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::reset_block,
                            Y[i]
                        );
                }
            }
        }

        global_X = global_X_ftr.get();

        /* do remaining SpMV computations */
        for (std::size_t i = 0; i != local_np; ++i)
        {
            for (std::size_t j = 0; j != np; ++j)
            {
                if(map_locality[j] != locality_id && A[i*np + j].nnz_ > 0)
                {
                    std::size_t buffer_target = (i*np + j) % num_threads;
                    Y_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::matvec_block,
                            A[i*np + j],
                            global_X[ map_locality[j] ][ map_block[j] ],
                            Y_buffer[buffer_target]
                        );
                    Y[i] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::accumulate_block,
                            Y[i], Y_buffer[buffer_target]
                        );
                    Y_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::reset_block,
                            Y[i]
                        );
                }
            }
        }
    
        // find local squared norm
        for (std::size_t i = 0; i != local_np; ++i)
        {
            sq_partial_norm_ftr[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::dot_block,
                    Y[i]
                );
        }

        // which is the sum of the partial sums that locality owns
        sq_local_norm_ftr = hpx::dataflow(
                [](std::vector<hpx::future<double>> norm_list) -> double
                {
                    double rv = 0.0;
                    std::size_t size = norm_list.size();
                    for (std::size_t i = 0; i != size; ++i)
                    {
                        rv += norm_list[i].get();
                    }
                    return rv;
                },
                std::move(sq_partial_norm_ftr)
        );

        // use all_reduce to compute the norm globally
        sq_global_norm_ftr = hpx::lcos::all_reduce(reduce_basename, std::move(sq_local_norm_ftr), std::plus<double>{}, nl, t);

        if (locality_id == 0)
        {
            norms[t] = sq_global_norm_ftr;
        }

        // normalize Y
        for (std::size_t i = 0; i != local_np; ++i)
        {
            Y[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::normalize_block,
                    Y[i], sq_global_norm_ftr
                );
        }

        // copy Y to X
        for (std::size_t i = 0; i != local_np; ++i)
        {
            X[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::copy_block,
                    Y[i], X[i]
                );
        }

    }

    if (locality_id == 0)
    {
        hpx::wait_all(norms);
        double time_passed = timer.elapsed();
        for (std::size_t t = 0; t <= nt; ++t)
        {
            printf("%.4lf", time_passed/nt);
            if (t != nt)
                printf(",");
            else
               printf("\n");
        }
        for (std::size_t t = 0; t < nt; ++t)
        {
            printf("%.4lf", sqrt(norms[t].get()));
            if (t != nt - 1)
                printf(",");
        }
        printf("\n");
    }

    delete[] map_locality;
    delete[] map_block;

    return Y;
}


///////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, block_data const& c)
{
    os << "{";
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << c[i];
    }
    os << "}";
    return os;
}


///////////////////////////////////////////////////////////////////////////////
void do_all_work(std::uint64_t block_size, std::uint64_t nt, 
        std::uint64_t buffer, std::string str_file)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();                    // Number of localities

    // Create the local iterate instance, register it
    iterate iteration(nl);

    // Measure execution time.
    std::uint64_t t = hpx::chrono::high_resolution_clock::now();

    // Perform all work and wait for it to finish
    hpx::future<iterate_server::Vec> result = iteration.do_work(nl, block_size, 
            nt, buffer, str_file);

    // Gather results from all localities
    if (hpx::get_locality_id() == 0)
    {
        std::uint64_t const num_worker_threads = hpx::get_num_worker_threads();

        hpx::future<std::vector<iterate_server::Vec> > overall_result =
            hpx::lcos::gather_here(gather_basename, std::move(result), nl);

        std::vector<iterate_server::Vec> solution = overall_result.get();
        for (std::size_t i = 0; i != nl; ++i)
        {
            iterate_server::Vec const& s = solution[i];
            for (std::size_t i = 0; i != s.size(); ++i)
            {
                s[i].get_data().wait();
            }
        }

        std::uint64_t elapsed = hpx::chrono::high_resolution_clock::now() - t;

        // Print the solution at time-iteration 'nt'.
        if (print_eigs)
        {
            for (std::size_t i = 0; i != nl; ++i)
            {
                iterate_server::Vec const& s = solution[i];
                for (std::size_t j = 0; j != s.size(); ++j)
                {
                    std::cout << "X[" << i*(s.size()) + j << "] = "
                        << s[j].get_data().get()
                        << std::endl;
                }
            }
        }

        std::cout << "Localities,OS_Threads,Execution_Time_sec,"
            "Block_Size,Iterations,SpMV_Buffer\n"
            << std::flush;
        std::cout << std::uint32_t(nl) << "," << num_worker_threads << ","
            << elapsed/1e9 << "," << block_size << "," << nt << "," << buffer << "\n";
    }
    else
    {
        hpx::lcos::gather_there(gather_basename, std::move(result)).wait();
    }
}



int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t block_size = vm["block_size"].as<std::uint64_t>();   // Local dimension (of each block).
    std::uint64_t nt = vm["nt"].as<std::uint64_t>();   // Number of solver iterations.
    std::uint64_t buffer = vm["buffer"].as<std::uint64_t>();   // SpMV output buffer count (times number of threads per rank).
    std::string str_file = vm["matrix_file"].as<std::string>();

    if (vm.count("results"))
        print_eigs = true;

    do_all_work(block_size, nt, buffer, str_file);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("results", "print the computed eigenvector (default: false)")
        ("block_size", value<std::uint64_t>()->default_value(10),
         "Local dimension (of each block)")
        ("nt", value<std::uint64_t>()->default_value(10),
         "Number of solver iterations")
        ("buffer", value<std::uint64_t>()->default_value(1),
         "SpMV output buffer count (times number of threads per rank)")
        ("matrix_file", value<std::string>()->default_value(""),
         "Binary custom matrix file")
    ;

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
