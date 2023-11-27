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
        ref_t<IShaderProgram> mVertexShader   = nullptr;
        ref_t<IShaderProgram> mFragmentShader = nullptr;
        ref_t<IRenderContext> RenderPass      = nullptr;
    };

    class EffectProcessor
    {
      public:
        EffectProcessor( ref_t<IGraphicContext> mGraphicContext, ref_t<IRenderContext> aRenderContext,
                         EffectProcessorCreateInfo aCreateInfo );
        ~EffectProcessor() = default;

        void Render( ref_t<ISampler2D> aImageSampler, ref_t<IRenderContext> aRenderContext );

        EffectProcessorCreateInfo   Spec;
        ref_t<IDescriptorSetLayout> PipelineLayout = nullptr;
        ref_t<IDescriptorSet>       mTextures      = nullptr;

      private:
        ref_t<IGraphicContext>   mGraphicContext    = nullptr;
        ref_t<IGraphicsPipeline> mPipeline          = nullptr;
        ref_t<IGraphicBuffer>    mCameraBuffer      = nullptr;
        ref_t<IDescriptorSet>    mCameraDescriptors = nullptr;
    };

} // namespace SE::Core