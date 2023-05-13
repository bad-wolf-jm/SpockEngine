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
        OptixModuleObject( const std::string aLaunchParameterVariableName, const char *aPtxCode,
                           Ref<OptixDeviceContextObject> aRayTracingContext );

        ~OptixModuleObject();

        OptixPipelineCompileOptions GetPipelineCompileOptions() { return mPipelineCompileOptions; }

        OptixPipelineLinkOptions GetPipelineLinkOptions() { return mPipelineLinkOptions; };

        void CreateMissGroup( std::string aEntryName );
        void CreateRayGenGroup( std::string aEntryName );
        void CreateHitGroup( std::string aClosestHitEntryName, std::string aAnyHitHitEntryName );

        Ref<OptixPipelineObject> CreatePipeline();

        std::vector<Ref<OptixProgramGroupObject>> mRayGenProgramGroups = {};
        std::vector<Ref<OptixProgramGroupObject>> mHitProgramGroups    = {};
        std::vector<Ref<OptixProgramGroupObject>> mMissProgramGroups   = {};

      private:
        Ref<OptixDeviceContextObject> mRayTracingContext = nullptr;
        OptixPipelineLinkOptions      mPipelineLinkOptions{};
        OptixPipelineCompileOptions   mPipelineCompileOptions{};
        const std::string             mLaunchParameterVariableName;
    };

} // namespace SE::Graphics