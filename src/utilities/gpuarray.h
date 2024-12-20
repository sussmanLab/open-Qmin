#ifndef GPUARRAY_H
#define GPUARRAY_H

/*!\file gpuarray.h */

//!A structure for declaring where we want to access data
struct access_location
    {
    //!An enumeration of possibilities
    enum Enum
        {
        host,   //!<We want to access the data on the CPU
        device  //!<We want to access the data on the GPU
        };
    };
//!A structure for declaring where the current version of the data is
struct data_location
    {
    //!An enumeration of possibilities
    enum Enum
        {
        host,       //!< data was last modified on host
        device,     //!< data was last modified on device
        hostdevice  //!< data is current on both host and device
        };
    };

//!A structure for declaring how we want to access data (read, write, overwrite?)
struct access_mode
    {
    //!An enumeration of possibilities
    enum Enum
        {
        read,       //!< we just want to read
        readwrite,  //!< we intend to both read and write
        overwrite   //!< we will completely overwrite all of the data
        };
    };


template<class T> class GPUArray;

template<class T> class ArrayHandle
    {
    public:
        //!the only constructor takes a reference to the GPUArray, a location and a mode
        ArrayHandle( GPUArray<T>& gpu_array,  access_location::Enum location = access_location::host,
                            access_mode::Enum mode = access_mode::readwrite);
        ~ArrayHandle();

        T* data;          //!< a pointer to the GPUArray's data

        void operator=( ArrayHandle& rhs)
                {
                data=rhs.data;
                };

    private:
         GPUArray<T>& gpu_array; //!< The GPUArray that the Handle was initialized with
    };

template<class T> class GPUArray
    {
    public:
        GPUArray();
        //! The most common constructor takes in the desired size of the array
        GPUArray(unsigned int num_elements, bool _register=false);
        virtual ~GPUArray();

        GPUArray( GPUArray& from);
        GPUArray& operator=( GPUArray& rhs);

        mutable data_location::Enum whereIsTheData;    //!< Tracks the current location of the data
        //!Swap two GPUArrays efficiently
        inline void swap(GPUArray& from);
        //!Get the size of the array
        unsigned int getNumElements()
            {
            return size();
            }
        unsigned int size() 
            {
            return arraySize;
            }
        //! Switch from simple memcpys to HostRegister pinned memory copies. Not currently fully functional
        //!Resize the array...performs operations on both the CPU and GPU
        virtual void resize(unsigned int num_elements);

    #ifdef ENABLE_CUDA 
        void setRegistered(bool _reg);
    #endif

    protected:
        inline void memclear(unsigned int first=0);

        inline T* acquire( access_location::Enum location,  access_mode::Enum mode) ;

        inline void release() 
            {
            Acquired = false;
            }

        mutable unsigned int arraySize;            //!< Number of elements
        mutable bool Acquired;                //!< Tracks whether the data has been acquired

        bool RegisterArray;                //!< Tracks whether the data has been acquired
        mutable bool dataHasTouchedDevice; //!< Tracks whether the data has ever been copied to the resizeDeviceArray
    
        mutable T* h_data; //!<pointer to memory on host

        inline void allocate();
        inline void deallocate();
        inline T* resizeHostArray(unsigned int num_elements);

#ifdef ENABLE_CUDA
        mutable T* d_data; //!<pointer to memory on device
        inline void memcpyDeviceToHost();
        inline void memcpyHostToDevice();
        inline T* resizeDeviceArray(unsigned int num_elements);
#endif

        friend class ArrayHandle<T>;
    };
#endif
