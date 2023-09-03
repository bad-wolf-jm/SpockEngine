#include "OptixProgramGroup.h"
#include "Core/CUDA/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixProgramGroupObject::OptixProgramGroupObject( OptixProgramGroupDesc         aProgramGroupDescription,
                                                      OptixProgramGroupOptions      aProgramGroupOptions,
                                                      ref_t<OptixDeviceContextObject> aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
    {

        char   lLogString[2048];
        size_t lLogStringSize = sizeof( lLogString );

        OPTIX_CHECK( optixProgramGroupCreate( mRayTracingContext->mOptixObject, &aProgramGroupDescription, 1, &aProgramGroupOptions,
                                              lLogString, &lLogStringSize, &mOptixObject ) );

        if( lLogStringSize > 1 ) SE::Logging::Info( "{}", lLogString );
    }

    OptixProgramGroupObject::~OptixProgramGroupObject() { OPTIX_CHECK_NO_EXCEPT( optixProgramGroupDestroy( mOptixObject ) ); }

} // namespace SE::Graphics