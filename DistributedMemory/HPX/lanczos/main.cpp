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

#include <mkl.h>

bool print_eigs = false;
char const* gather_basename = "/dist_lanczos/gather/";
char const* reduce_basename_norm = "/dist_lanczos/reduce/norm/";
char const* reduce_basename_dotp = "/dist_lanczos/reduce/dotp/";
char const* reduce_basename_xty  = "/dist_lanczos/reduce/xty/";
char const* all_gather_basename_spmv = "/dist_power_iteration/all_gather/spmv/";
char const* all_gather_basename_xty  = "/dist_power_iteration/all_gather/xty/";


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
// a class that holds data for each 2D block, that is, a chunk of tall-skinny matrix
struct block2D_data
{
private:
    typedef hpx::serialization::serialize_buffer<double> buffer_type;

public:
    block2D_data()
      : size_(0), width_(0)
    {}

    // Create a new (uninitialized) block of the given size.
    explicit block2D_data(std::size_t size, std::size_t width)
      : data_(std::allocator<double>().allocate(size*width), size*width, buffer_type::take),
        size_(size), width_(width)
    {}

    // Create a new (initialized) block of the given size.
    block2D_data(std::size_t size, std::size_t width, double initial_value)
      : data_(std::allocator<double>().allocate(size*width), size*width, buffer_type::take),
        size_(size), width_(width)
    {
        for (std::size_t i = 0; i != size*width; ++i)
            data_[i] = initial_value;
    }

    double& operator[](std::size_t idx) { return data_[idx]; }
    double operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return size_; }
    std::size_t width() const { return width_; }

private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & data_ & size_ & width_;
        // clang-format on
    }

private:
    buffer_type data_;
    std::size_t size_, width_;
};

///////////////////////////////////////////////////////////////////////////////
// a class that holds data for each matrix block in CSB format
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
// This is the server side representation of the 2D block data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
struct block2D_server
  : hpx::components::component_base<block2D_server>
{   
    // construct new instances
    block2D_server() = default;
    
    explicit block2D_server(block2D_data const& data)
      : data_(data)
    {}
    
    block2D_server(std::size_t size, std::size_t width, double initial_value)
      : data_(size, width, initial_value)
    {}
    
    block2D_data get_data() const
    {   
        return data_;
    }
    
    // Every member function which has to be invoked remotely needs to be
    // wrapped into a component action. The macro below defines a new type
    // 'get_data_action' which represents the (possibly remote) member function
    // block2D::get_data().
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(block2D_server, get_data, get_data_action);

private:
    block2D_data data_;
};

// The macros below are necessary to generate the code required for exposing
// our 2D block type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<block2D_server> block2D_server_type;
HPX_REGISTER_COMPONENT(block2D_server_type, block2D_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef block2D_server::get_data_action get_2Ddata_action;
HPX_REGISTER_ACTION(get_2Ddata_action);


///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate while referencing a remote 2D block.
struct block2D : hpx::components::client_base<block2D, block2D_server>
{
    typedef hpx::components::client_base<block2D, block2D_server> base_type;

    block2D() = default;

    // Create new component on locality 'where' and initialize the held data
    block2D(hpx::id_type where, std::size_t size, std::size_t width, double initial_value)
      : base_type(hpx::new_<block2D_server>(where, size, width, initial_value))
    {}

    // Create a new component on the locality co-located to the id 'where'. The
    // new instance will be initialized from the given block2D_data.
    block2D(hpx::id_type where, block2D_data const& data)
      : base_type(hpx::new_<block2D_server>(hpx::colocated(where), data))
    {}

    // Attach a future representing a (possibly remote) block2D.
    block2D(hpx::future<hpx::id_type> && id) noexcept
      : base_type(std::move(id))
    {}

    // Unwrap a future<block2D> (a block2D already is a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    block2D(hpx::future<block2D> && c) noexcept
      : base_type(std::move(c))
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<block2D_data> get_data() const
    {
        block2D_server::get_data_action act;
        return hpx::async(act, get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
// Data for each iteration on one locality
struct iterate_server : hpx::components::component_base<iterate_server>
{
    // Our data for each vector
    typedef std::vector<block> Vec;
    typedef std::vector<block2D> Vec2D;
    typedef std::vector<matrix_block> Mat;

    // Construct new instances
    iterate_server() = default;

    explicit iterate_server(std::size_t nl)
    {}

    // Do all the work on 'np' blocks, 'nx' data points each, for 'nt'
    // iterations, limit depth of dependency tree to 'nd'.
    Vec2D do_work(std::size_t nl, std::size_t block_size, std::size_t nt, 
            std::size_t eig_wanted, std::size_t buffer, std::string str_file);

    HPX_DEFINE_COMPONENT_ACTION(iterate_server, do_work, do_work_action);

protected:
    static void read_custom(int &numrows, int &numcols, int &nnonzero,
            char* filename, int *&colptrs, int *&irem, double *&xrem, 
            int nl, int block_size);
    static void csc2blkcoord(iterate_server::Mat &matrixBlock,
            int *colptrs, int *irem, double *xrem, int numcols,
            int base_rowblk, int nrowblks, int ncolblks, int block_size);
    static block reset_block(block const& V);
    static block2D copy_block(block const& src, block2D const& dst, int offset);
    static block accumulate_block(block const& src, block const& dst);
    static block subtract_block(block const& src, block const& dst);
    static block matvec_block(matrix_block const& A, block const& X, block const& Y);
    static double norm_block(block const& V);
    static double dotp_block(block const& V1, block const& V2);
    static std::vector<double> xty_block(block2D const& V1, block const& V2, int avail_eigs);
    static block xy_block(block2D const& V1, block const& V2, int avail_eigs);
    static block normalize_block(block const& V, hpx::shared_future<double> norm_ftr);

private:
    block QpZ;
    Vec qq, z, z_buffer, QQpZ;
    Vec2D Q;
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

    hpx::future<iterate_server::Vec2D> do_work(std::size_t nl,
            std::size_t block_size, std::size_t nt, std::size_t eig_wanted,
            std::size_t buffer, std::string str_file)
    {
        return hpx::async(do_work_action(), get_id(), nl, block_size, nt, 
                eig_wanted, buffer, str_file);
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
block2D iterate_server::copy_block(block const& src, block2D const& dst, int offset)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [src, dst, offset](block_data const& src_avail, block2D_data dst_avail) -> block2D
            {
                HPX_UNUSED(src);       

                std::size_t size = dst_avail.size();
                std::size_t width = dst_avail.width();
                for (std::size_t i = 0; i != size; ++i)
                {
                    dst_avail[i*width+offset] = src_avail[i];
                }

                return block2D(dst.get_id(), dst_avail);
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
block iterate_server::subtract_block(block const& src, block const& dst)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [src, dst](block_data const& src_avail, block_data dst_avail) -> block
            {
                //TODO: maybe remove this and that above
                HPX_UNUSED(src);       

                std::size_t size = dst_avail.size();
                for (std::size_t i = 0; i != size; ++i)
                {
                    dst_avail[i] -= src_avail[i];
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
double iterate_server::norm_block(block const& V)
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
double iterate_server::dotp_block(block const& V1, block const& V2)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [V1, V2](block_data const& V1_avail, block_data V2_avail) -> double
            {
                double rv = 0.0;
                std::size_t size = V1_avail.size();
                for (std::size_t i = 0; i != size; ++i)
                {
                    rv += V1_avail[i] * V2_avail[i];
                }

                return rv;
            }
        ),
        V1.get_data(),
        V2.get_data()
   ).get();
}

///////////////////////////////////////////////////////////////////////////////
std::vector<double> iterate_server::xty_block(block2D const& V1, block const& V2, int avail_eigs)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [V1, V2, avail_eigs](block2D_data const& V1_avail, block_data V2_avail) -> std::vector<double>
            {
                std::size_t size = V1_avail.size();
                std::size_t width = V1_avail.width();
                std::vector<double> rv(width, 0.0);
                //TODO: dgemv_t
                for (std::size_t i = 0; i != size; ++i)
                {
                    for (std::size_t j = 0; j != avail_eigs; ++j)
                    {
                        rv[j] += V1_avail[i*width+j] * V2_avail[i];
                    }
                }

                return rv;
            }
        ),
        V1.get_data(),
        V2.get_data()
   ).get();
}


///////////////////////////////////////////////////////////////////////////////
block iterate_server::xy_block(block2D const& V1, block const& V2, int avail_eigs)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::util::unwrapping(
            [V1, V2, avail_eigs](block2D_data const& V1_avail, block_data V2_avail) -> block
            {
                std::size_t size = V1_avail.size();
                std::size_t width = V1_avail.width();
                block_data next(size, 0.0);
                //TODO: dgemv
                for (std::size_t i = 0; i != size; ++i)
                {
                    for (std::size_t j = 0; j != avail_eigs; ++j)
                    {
                        next[i] += V1_avail[i*width+j] * V2_avail[j];
                    }
                }

                return block(V1.get_id(), next);
            }
        ),
        V1.get_data(),
        V2.get_data()
   );
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

std::ostream& operator<<(std::ostream& os, block_data const& c);

///////////////////////////////////////////////////////////////////////////////
// This is the implementation of main loop
//
// Do the work on 'np/nl' blocks, 'nx' data points each, for 'nt' iterations.
iterate_server::Vec2D iterate_server::do_work(std::size_t nl,
        std::size_t block_size, std::size_t nt, std::size_t eig_wanted,
        std::size_t buffer, std::string str_file)
{
    nt = eig_wanted;
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
    qq.resize(local_np);
    z.resize(local_np);
    z_buffer.resize(num_threads);
    QQpZ.resize(local_np);
    Q.resize(local_np);
    A.resize(local_np * np);
    
    // convert the matrix to csb format
    csc2blkcoord(A, colptrs, irem, xrem, dim, base_rowblk, local_np, np, block_size);

    double *alpha = new double[eig_wanted];
    double *beta = new double[eig_wanted];
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

    hpx::id_type here = hpx::find_here();

    // QpZ = zeros(RHS)
    QpZ = block(here, eig_wanted, 0.0);

    // qq = ones(numcols)/||ones(numcols)||
    for (std::size_t i = 0; i != local_np; ++i)
    {
        if (locality_id != nl-1 || i != local_np-1)
        {
            qq[i] = block(here, block_size, 1.0/sqrt(dim));
        }
        else
        {
            block_data last_one(block_size, 1.0/sqrt(dim));
            for (std::size_t j = last_block; j != block_size; ++j)
            {
                last_one[j] = 0;
            }
            qq[i] = block(here, last_one);
        }
    }


    // z = zeros(numcols)
    for (std::size_t i = 0; i != local_np; ++i)
    {
        z[i] = block(here, block_size, 0.0);
    }

    // z_buffer = zeros(buffer_size * block_size)
    for (std::size_t i = 0; i != num_threads; ++i)
    {
        z_buffer[i] = block(here, block_size, 0.0);
    }

    // QQpZ = zeros(numcols)
    for (std::size_t i = 0; i != local_np; ++i)
    {
        QQpZ[i] = block(here, block_size, 0.0);
    }

    // Q = zeros(numcols, eig_wanted)
    for (std::size_t i = 0; i != local_np; ++i)
    {
        Q[i] = block2D(here, block_size, eig_wanted, 0.0);
    }

    // Q[column 0] = qq
    for (std::size_t i = 0; i != local_np; ++i)
    {
        Q[i] = hpx::dataflow(
                hpx::launch::async, &iterate_server::copy_block,
                qq[i], Q[i], 0
            );
    }

    hpx::wait_all(QpZ);
    hpx::wait_all(qq);
    hpx::wait_all(z);
    hpx::wait_all(z_buffer);
    hpx::wait_all(QQpZ);
    hpx::wait_all(Q);

    std::vector<hpx::future<double>> partial_norm_ftr(local_np);
    std::vector<hpx::future<double>> partial_dotp_ftr(local_np);
    std::vector<hpx::future<std::vector<double>>> partial_xty_ftr(local_np);

    hpx::future<double> local_norm_ftr;
    hpx::future<double> local_dotp_ftr;
    hpx::future<std::vector<double>> local_xty_ftr;

    hpx::shared_future<double> global_norm_ftr;
    hpx::shared_future<double> global_dotp_ftr;
    hpx::future<std::vector<std::vector<double>>> global_xty_ftr;

    std::vector<hpx::shared_future<double>> alpha_ftr (nt);
    std::vector<hpx::shared_future<double>> beta_ftr  (nt);
    hpx::future<std::vector<iterate_server::Vec>> global_qq_ftr;
    std::vector<iterate_server::Vec> global_qq;

    hpx::chrono::high_resolution_timer timer;

    for (std::size_t t = 0; t != nt; ++t)
    {
        // z = 0
        for (std::size_t i = 0; i != local_np; ++i)
        {   
            // reset ith block of z when ith block of qq is ready
            // to set the dependency between subsequent iterations
            z[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::reset_block,
                    qq[i]
                );
        }

        // z = A*qq

        // all_gather qq partitions/blocks from all localities
        // hpx::future<std::vector<iterate_server::Vec> > global_qq =
        global_qq_ftr = hpx::lcos::all_gather(all_gather_basename_spmv, qq, nl, t);

        /* do local SpMV computations using local qq blocks */
        for (std::size_t i = 0; i != local_np; ++i)
        {   
            for (std::size_t j = 0; j != local_np; ++j)
            {   
                if(A[i*np + j + base_rowblk].nnz_ > 0)
                {
                    std::size_t buffer_target = (i*np + j + base_rowblk) % num_threads;
                    z_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::matvec_block,
                            A[i*np + j + base_rowblk],
                            qq[j],
                            z_buffer[buffer_target]
                        );
                    z[i] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::accumulate_block,
                            z[i], z_buffer[buffer_target]
                        );
                    z_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::reset_block,
                            z[i]
                        );
                }
            }
        }

        global_qq = global_qq_ftr.get();

        /* do remaining SpMV computations */
        for (std::size_t i = 0; i != local_np; ++i)
        {
            for (std::size_t j = 0; j != np; ++j)
            {
                if(map_locality[j] != locality_id && A[i*np + j].nnz_ > 0)
                {
                    std::size_t buffer_target = (i*np + j) % num_threads;
                    z_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::matvec_block,
                            A[i*np + j],
                            global_qq[ map_locality[j] ][ map_block[j] ],
                            z_buffer[buffer_target]
                        );
                    z[i] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::accumulate_block,
                            z[i], z_buffer[buffer_target]
                        );
                    z_buffer[buffer_target] = hpx::dataflow(
                            hpx::launch::async, &iterate_server::reset_block,
                            z[i]
                        );
                }
            }
        }

        // alpha[t] = qq'z

        // find local dot product
        for (std::size_t i = 0; i != local_np; ++i)
        {
            partial_dotp_ftr[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::dotp_block,
                    qq[i], z[i]
                );
        }

        // which is the sum of the partial dot products that locality owns
        local_dotp_ftr = hpx::dataflow(
                [](std::vector<hpx::future<double>> dotp_list) -> double
                {
                    double rv = 0.0;
                    std::size_t size = dotp_list.size();
                    for (std::size_t i = 0; i != size; ++i)
                    {
                        rv += dotp_list[i].get();
                    }
                    return rv;
                },
                std::move(partial_dotp_ftr)
        );

        // use all_reduce to compute the norm globally
        global_dotp_ftr = hpx::lcos::all_reduce(reduce_basename_dotp, std::move(local_dotp_ftr), std::plus<double>{}, nl, t);

        if (locality_id == 0)
        {
            alpha_ftr[t] = global_dotp_ftr;
        }

        // QpZ = Q'z
        // QpZ[RHS x 1] RHS = eigs computed = number of iterations in lanczos
        // Q[matrix_dim x RHS] --> set of vectors from each iteration
        // z[matrix_dim x 1]
        
        // find local XTY result first
        for (std::size_t i = 0; i != local_np; ++i)
        {
            partial_xty_ftr[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::xty_block,
                    Q[i], z[i], t+1
                );
        }

        // which is the sum of the partial XTY results that locality owns
        local_xty_ftr = hpx::dataflow(
                [eig_wanted](std::vector<hpx::future<std::vector<double>>> partial_xty_list) -> std::vector<double>
                {
                    std::vector<double> local_xty(eig_wanted, 0.0);
                    std::size_t size = partial_xty_list.size();
                    for (std::size_t i = 0; i != size; ++i)
                    {
                        std::vector<double> partial_xty = partial_xty_list[i].get();
                        for (std::size_t j = 0; j != eig_wanted; ++j)
                        {
                            local_xty[j] += partial_xty[j];
                        }
                    }
                    return local_xty;
                },
                std::move(partial_xty_ftr)
        );

        // use all_reduce to compute the XTY globally
        // global_xty_ftr = hpx::lcos::all_reduce(reduce_basename_xty, std::move(local_xty_ftr), std::plus<std::vector<double>>{}, nl, t);
        // 
        // all_reduce does not seem to work on an array, that is,
        // the number of elements in send buffer should be equal to one 
        // (imagine having to set count to 1 in MPI's all_reduce operation)
        // therefore, we have to settle with all_gather instead
        // which should not be too terrible given that each locality will get
        // O(num_localities * RHS) elements instead of O(RHS).
        global_xty_ftr = hpx::lcos::all_gather(all_gather_basename_xty, std::move(local_xty_ftr), nl, t);
        
        // whenever all_gather operation above completes, compute QpZ,
        // which requires manual reduction
        QpZ = hpx::dataflow(
                [here](std::vector<std::vector<double>> local_xty_list) -> block
                {
                    std::size_t size1 = local_xty_list.size();
                    std::size_t size2 = local_xty_list[0].size();
                    block_data global_xty(size2, 0.0);
                    for (std::size_t i = 0; i != size1; ++i)
                    {
                        for (std::size_t j = 0; j != size2; ++j)
                        {
                            global_xty[j] += local_xty_list[i][j];
                        }
                    }

                    return block(here, global_xty);
                },
                std::move(global_xty_ftr.get())
        );

        // QQpZ = Q*QpZ
        // QQpZ[matrix_dim x 1]
        // Q[matrix_dim x RHS]
        // QPZ[RHS X 1]
        for (std::size_t i = 0; i != local_np; ++i)
        {
            QQpZ[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::xy_block,
                    Q[i], QpZ, t+1
                );
        }

        // z = z - QQpZ
        for (std::size_t i = 0; i != local_np; ++i)
        {
            z[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::subtract_block,
                    QQpZ[i], z[i]
                );
        }

        // beta[t] = ||z||

        // find local squared norm
        for (std::size_t i = 0; i != local_np; ++i)
        {
            partial_norm_ftr[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::norm_block,
                    z[i]
                );
        }

        // which is the sum of the partial sums that locality owns
        local_norm_ftr = hpx::dataflow(
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
                std::move(partial_norm_ftr)
        );

        // use all_reduce to compute the norm globally
        global_norm_ftr = hpx::lcos::all_reduce(reduce_basename_norm, std::move(local_norm_ftr), std::plus<double>{}, nl, t);

        if (locality_id == 0)
        {
            beta_ftr[t] = global_norm_ftr;
        }

        // qq = z/||z||
        for (std::size_t i = 0; i != local_np; ++i)
        {
            qq[i] = hpx::dataflow(
                    hpx::launch::async, &iterate_server::normalize_block,
                    z[i], global_norm_ftr
                );
        }

        // Q[column t+1] = qq
        if (t != eig_wanted-1)
        {
            for (std::size_t i = 0; i != local_np; ++i)
            {
                Q[i] = hpx::dataflow(
                        hpx::launch::async, &iterate_server::copy_block,
                        qq[i], Q[i], t+1
                    );
            }
        }
    }

    hpx::wait_all(alpha_ftr);
    hpx::wait_all(beta_ftr);
    hpx::wait_all(Q);

    if (locality_id == 0)
    {
        for (std::size_t t = 0; t != eig_wanted; ++t)
        {
            alpha[t] = alpha_ftr[t].get();
            beta[t] = sqrt(beta_ftr[t].get());
        }

        double time_passed = timer.elapsed();
        
        for (std::size_t t = 0; t <= eig_wanted; ++t)
        {
            printf("%.4lf", time_passed/nt);
            if (t != eig_wanted)
                printf(",");
        }
        printf("\n");
        
        for (std::size_t t = 0; t != eig_wanted; ++t)
        {
            printf("%.4lf", alpha[t]);
            if (t != eig_wanted - 1)
                printf(",");
        }
        printf("\n");
        
        for (std::size_t t = 0; t != eig_wanted; ++t)
        {
            printf("%.4lf", beta[t]);
            if (t != eig_wanted - 1)
                printf(",");
        }
        printf("\n");
        
        LAPACKE_dsterf(eig_wanted, alpha, beta);
        
        for (std::size_t t = 0; t != eig_wanted; ++t)
        {
            printf("%.4lf", alpha[t]);
            if (t != eig_wanted - 1)
                printf(",");
        }
        printf("\n");
    }

    delete[] map_locality;
    delete[] map_block;

    return Q;
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
std::ostream& operator<<(std::ostream& os, block2D_data const& c)
{
    int size = c.size();
    int width = c.width();
    int index;

    os << "{\n";
    for (std::size_t i = 0; i != size; ++i)
    {
        for (std::size_t j = 0; j != width; ++j)
        {
            index = i*width+j;
            if (index != 0)
                os << ", ";
            os << c[index];
        }
        os << "\n";
    }
    os << "}\n";
    return os;
}


///////////////////////////////////////////////////////////////////////////////
void do_all_work(std::uint32_t block_size, std::uint32_t nt, std::uint32_t eig_wanted,
        std::uint32_t buffer, std::string str_file)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();                    // Number of localities

    // Create the local iterate instance, register it
    iterate iteration(nl);

    // Measure execution time.
    std::uint64_t t = hpx::chrono::high_resolution_clock::now();

    // Perform all work and wait for it to finish
    hpx::future<iterate_server::Vec2D> result = iteration.do_work(nl, block_size, 
            nt, eig_wanted, buffer, str_file);

    // Gather results from all localities
    if (hpx::get_locality_id() == 0)
    {
        std::uint64_t const num_worker_threads = hpx::get_num_worker_threads();

        hpx::future<std::vector<iterate_server::Vec2D> > overall_result =
            hpx::lcos::gather_here(gather_basename, std::move(result), nl);

        std::vector<iterate_server::Vec2D> solution = overall_result.get();
        for (std::size_t i = 0; i != nl; ++i)
        {
            iterate_server::Vec2D const& s = solution[i];
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
                iterate_server::Vec2D const& s = solution[i];
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
    std::uint32_t block_size = vm["block_size"].as<std::uint32_t>();   // Local dimension (of each block).
    std::uint32_t nt = vm["nt"].as<std::uint32_t>();   // Number of solver iterations.
    std::uint32_t eig_wanted = vm["eig_wanted"].as<std::uint32_t>();   // RHS
    std::uint32_t buffer = vm["buffer"].as<std::uint32_t>();   // SpMV output buffer count (times number of threads per rank).
    std::string str_file = vm["matrix_file"].as<std::string>();

    if (vm.count("results"))
        print_eigs = true;

    do_all_work(block_size, nt, eig_wanted, buffer, str_file);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("results", "print the computed eigenvector (default: false)")
        ("block_size", value<std::uint32_t>()->default_value(65536),
         "Local dimension (of each block)")
        ("nt", value<std::uint32_t>()->default_value(10),
         "Number of solver iterations")
        ("eig_wanted", value<std::uint32_t>()->default_value(8),
         "RHS")
        ("buffer", value<std::uint32_t>()->default_value(1),
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
