#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/VertexData.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct DeferredLightingRendererCreateInfo
    {
        bool Opaque = false;

        Ref<IRenderContext> RenderPass = nullptr;
    };

    class DeferredLightingRenderer
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mNumSamples;
        };

        Ref<IDescriptorSetLayout> CameraSetLayout            = nullptr;
        Ref<IDescriptorSetLayout> TextureSetLayout           = nullptr;
        Ref<IDescriptorSetLayout> DirectionalShadowSetLayout = nullptr;
        Ref<IDescriptorSetLayout> SpotlightShadowSetLayout   = nullptr;
        Ref<IDescriptorSetLayout> PointLightShadowSetLayout  = nullptr;

      public:
        DeferredLightingRenderer() = default;
        DeferredLightingRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetDirectionalShadowSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext );

        Ref<IGraphicsPipeline> Pipeline() { return mPipeline; }

        ~DeferredLightingRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer   = nullptr;
        Ref<IGraphicsPipeline> mPipeline       = nullptr;
    };

} // namespace SE::Core