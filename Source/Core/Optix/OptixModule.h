#pragma once

#include "Core/Memory.h"
#include <string>
#include <vector>

#include "Optix7.h"
#include "OptixContext.h"
#include "OptixPipeline.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Cuda;

    struct OptixModuleObject
    {
        OptixModule mOptixObject = nullptr;

        OptixModuleObject() = default;
        OptixModuleObject( const string_t aLaunchParameterVariableName, const char *aPtxCode,
                           ref_t<OptixDeviceContextObject> aRayTracingContext );

        ~OptixModuleObject();

        OptixPipelineCompileOptions GetPipelineCompileOptions() { return mPipelineCompileOptions; }

        OptixPipelineLinkOptions GetPipelineLinkOptions() { return mPipelineLinkOptions; };

        void CreateMissGroup( string_t aEntryName );
        void CreateRayGenGroup( string_t aEntryName );
        void CreateHitGroup( string_t aClosestHitEntryName, string_t aAnyHitHitEntryName );

        ref_t<OptixPipelineObject> CreatePipeline();

        vec_t<ref_t<OptixProgramGroupObject>> mRayGenProgramGroups = {};
        vec_t<ref_t<OptixProgramGroupObject>> mHitProgramGroups    = {};
        vec_t<ref_t<OptixProgramGroupObject>> mMissProgramGroups   = {};

      private:
        ref_t<OptixDeviceContextObject> mRayTracingContext = nullptr;
        OptixPipelineLinkOptions      mPipelineLinkOptions{};
        OptixPipelineCompileOptions   mPipelineCompileOptions{};
        const string_t             mLaunchParameterVariableName;
    };

} // namespace SE::Graphics