#pragma once

#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/VertexData.h"

#include "Graphics/API.h"

namespace SE::Core
{

    using namespace SE::Graphics;

    struct EffectProcessorCreateInfo
    {
        Ref<IShaderProgram> mVertexShader   = "";
        Ref<IShaderProgram> mFragmentShader = "";

        Ref<IRenderContext> RenderPass = nullptr;
    };

    class EffectProcessor
    {
      public:
        EffectProcessor( Ref<IGraphicContext> mGraphicContext, Ref<IRenderContext> aRenderContext,
                         EffectProcessorCreateInfo aCreateInfo );
        ~EffectProcessor() = default;

        void Render( Ref<ISampler2D> aImageSampler, Ref<IRenderContext> aRenderContext );

        EffectProcessorCreateInfo Spec;
        Ref<IDescriptorSetLayout> PipelineLayout = nullptr;
        Ref<IDescriptorSet>       mTextures      = nullptr;

      private:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
    };

} // namespace SE::Core