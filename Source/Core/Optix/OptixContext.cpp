#include "OptixContext.h"
#include "Core/Cuda/CudaAssert.h"

namespace SE::Graphics
{

    OptixDeviceContextObject::OptixDeviceContextObject()
    {
        CUcontext lCudaContext;
        CUDA_ASSERT( cuCtxGetCurrent( &lCudaContext ) );
        OPTIX_CHECK( optixDeviceContextCreate( lCudaContext, 0, &mOptixObject ) );
    }

    OptixDeviceContextObject::OptixDeviceContextObject( CUcontext aCudaContext )
    {
        OPTIX_CHECK( optixDeviceContextCreate( aCudaContext, 0, &mOptixObject ) );
    }

    OptixDeviceContextObject::~OptixDeviceContextObject() { OPTIX_CHECK( optixDeviceContextDestroy( mOptixObject ) ); }

    void OptixDeviceContextObject::Initialize()
    {
        cudaFree( 0 );

        int lCudaDeviceCount;
        cudaGetDeviceCount( &lCudaDeviceCount );
        
        if( lCudaDeviceCount == 0 ) throw std::runtime_error( "No CUDA capable devices found!" );

        OPTIX_CHECK( optixInit() );
    }

} // namespace SE::Graphics