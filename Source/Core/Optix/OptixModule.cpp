#include "OptixModule.h"
#include "Core/CUDA/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixModuleObject::OptixModuleObject( const string_t aLaunchParameterVariableName, const char *aPtxCode,
                                          ref_t<OptixDeviceContextObject> aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
        , mLaunchParameterVariableName{ aLaunchParameterVariableName }
    {
        OptixModuleCompileOptions lModuleCompileOptions{};

        lModuleCompileOptions.maxRegisterCount = 50;
        lModuleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        lModuleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        mPipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        mPipelineCompileOptions.usesMotionBlur                   = false;
        mPipelineCompileOptions.numPayloadValues                 = 2;
        mPipelineCompileOptions.numAttributeValues               = 2;
        mPipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        mPipelineCompileOptions.pipelineLaunchParamsVariableName = mLaunchParameterVariableName.c_str();

        mPipelineLinkOptions.maxTraceDepth = 2;

        const string_t ptxCode = aPtxCode;

        char   lLogString[2048];
        size_t lLogStringSize = sizeof( lLogString );
        OPTIX_CHECK( optixModuleCreateFromPTX( mRayTracingContext->mOptixObject, &lModuleCompileOptions, &mPipelineCompileOptions,
                                               ptxCode.c_str(), ptxCode.size(), lLogString, &lLogStringSize, &mOptixObject ) );
        if( lLogStringSize > 1 ) SE::Logging::Info( "{}", lLogString );
    }

    OptixModuleObject::~OptixModuleObject() { OPTIX_CHECK_NO_EXCEPT( optixModuleDestroy( mOptixObject ) ); }

    void OptixModuleObject::CreateMissGroup( string_t aEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        lProgramGroupDescription.miss.module            = mOptixObject;
        lProgramGroupDescription.miss.entryFunctionName = aEntryName.c_str();

        mMissProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    void OptixModuleObject::CreateRayGenGroup( string_t aEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        lProgramGroupDescription.raygen.module            = mOptixObject;
        lProgramGroupDescription.raygen.entryFunctionName = aEntryName.c_str();

        mRayGenProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    void OptixModuleObject::CreateHitGroup( string_t aClosestHitEntryName, string_t aAnyHitHitEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
        lProgramGroupDescription.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        lProgramGroupDescription.hitgroup.moduleCH            = mOptixObject;
        lProgramGroupDescription.hitgroup.entryFunctionNameCH = aClosestHitEntryName.c_str();
        lProgramGroupDescription.hitgroup.moduleAH            = mOptixObject;
        lProgramGroupDescription.hitgroup.entryFunctionNameAH = aAnyHitHitEntryName.c_str();

        mHitProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    ref_t<OptixPipelineObject> OptixModuleObject::CreatePipeline()
    {
        vector_t<ref_t<OptixProgramGroupObject>> lProgramGroups;
        for( auto pg : mRayGenProgramGroups ) lProgramGroups.push_back( pg );
        for( auto pg : mHitProgramGroups ) lProgramGroups.push_back( pg );
        for( auto pg : mMissProgramGroups ) lProgramGroups.push_back( pg );
        return SE::Core::New<OptixPipelineObject>( mPipelineLinkOptions, mPipelineCompileOptions, lProgramGroups, mRayTracingContext );
    }

} // namespace SE::Graphics