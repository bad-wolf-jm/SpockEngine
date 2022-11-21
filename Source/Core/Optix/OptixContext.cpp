#include "OptixContext.h"
#include "Core/Cuda/CudaAssert.h"

namespace SE::Graphics
{

    OptixDeviceContextObject::OptixDeviceContextObject()
    {
        CUcontext l_CudaContext;
        CUDA_ASSERT( cuCtxGetCurrent( &l_CudaContext ) );
        OPTIX_CHECK( optixDeviceContextCreate( l_CudaContext, 0, &RTObject ) );
    }

    OptixDeviceContextObject::OptixDeviceContextObject( CUcontext a_CudaContext )
    {
        OPTIX_CHECK( optixDeviceContextCreate( a_CudaContext, 0, &RTObject ) );
    }

    OptixDeviceContextObject::~OptixDeviceContextObject() { OPTIX_CHECK( optixDeviceContextDestroy( RTObject ) ); }

    void OptixDeviceContextObject::Initialize()
    {
        cudaFree( 0 );
        int numDevices;
        cudaGetDeviceCount( &numDevices );
        if( numDevices == 0 ) throw std::runtime_error( "No CUDA capable devices found!" );

        OPTIX_CHECK( optixInit() );
    }

} // namespace SE::Graphics