#pragma once

#include "Core/GraphicContext/ARenderContext.h"
#include "Core/GraphicContext/ARenderTarget.h"
#include "Core/GraphicContext/DescriptorSet.h"
#include "Core/Memory.h"

#include "ASceneRenderer.h"

namespace LTSE::Core
{
    class DeferredRenderer : public ASceneRenderer
    {
      public:
        DeferredRenderer() = default;
        DeferredRenderer( GraphicContext aGraphicContext, eColorFormat aOutputFormat, uint32_t aOutputSampleCount );

        ~DeferredRenderer() = default;

        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

        void Update( Ref<Scene> aWorld );
        void Render();

      private:
        ARenderContext mGeometryContext{};
        ARenderContext mLightingContext{};

        Ref<ARenderTarget> mGeometryRenderTarget = nullptr;

        Ref<DescriptorSetLayout> mLightingTextureLayout = nullptr;
        Ref<DescriptorSet>       mLightingPassTextures  = nullptr;
        Ref<DescriptorSetLayout> mLightingCameraLayout  = nullptr;
        Ref<DescriptorSet>       mLightingPassCamera    = nullptr;
        Ref<ARenderTarget>       mLightingRenderTarget  = nullptr;
    };
} // namespace LTSE::Core