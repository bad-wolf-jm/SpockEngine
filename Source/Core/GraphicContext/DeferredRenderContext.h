#pragma once

#include "Core/GraphicContext//GraphicContext.h"

#include "GraphicsPipeline.h"
#include "RenderTarget.h"

namespace LTSE::Graphics
{

    class DeferredRenderContext
    {
      public:
        DeferredRenderContext() = default;
        DeferredRenderContext( GraphicContext const &aGraphicContext, Ref<AbstractRenderTarget> aRenderTarget, Ref<AbstractRenderTarget> aLightingRenderTarget );

        ~DeferredRenderContext() = default;

        GraphicContext &GetGraphicContext() { return mGraphicContext; };

        uint32_t                              GetOutputImageCount();
        Ref<AbstractRenderTarget>             GetRenderTarget() { return mRenderTarget; }
        Ref<Internal::sVkCommandBufferObject> GetCurrentCommandBuffer()
        {
            return mRenderTarget->GetCommandBuffer( mCurrentCommandBuffer );
        }
        Ref<Internal::sVkDeferredRenderPassObject> GetRenderPass() { return mRenderPass; }
        Ref<Internal::sVkLightingRenderPassObject> GetLightingRenderPass() { return mLightingRenderPass; }

        bool BeginRender();
        bool EndRender();

        void Bind( Ref<GraphicsPipeline> a_GraphicPipeline );
        void Bind( Ref<DescriptorSet> a_DescriptorSet, uint32_t a_SetIndex, int32_t a_DynamicOffset = -1 );
        void Bind( Ref<Buffer> a_VertexBuffer, uint32_t a_BindPoint = 0 );
        void Bind( Ref<Buffer> a_VertexBuffer, Ref<Buffer> a_IndexBuffer, uint32_t a_BindPoint = 0 );

        template <typename T>
        void PushConstants( Internal::ShaderStageType a_ShaderStages, uint32_t a_Offset, const T &a_Value )
        {
            auto lCommandBuffer = mRenderTarget->GetCommandBuffer( mCurrentCommandBuffer );
            lCommandBuffer->PushConstants<T>( (VkShaderStageFlags)a_ShaderStages, a_Offset, a_Value, mCurrentPipelineLayout );
        }

        void Draw( uint32_t a_VertexCount, uint32_t a_VertexOffset, uint32_t a_VertexBufferOffset = 0, uint32_t a_InstanceCount = 0,
            uint32_t a_FirstInstance = 0 );

        void Present();

        void ResetBuffers();

        operator bool() { return static_cast<bool>( mRenderTarget ) && static_cast<bool>( mRenderPass ) && static_cast<bool>( mLightingRenderPass ); }

      private:
        GraphicContext mGraphicContext{};

        bool mFrameIsStarted = false;

        uint32_t mCurrentCommandBuffer = 0;

        Ref<AbstractRenderTarget> mRenderTarget = nullptr;

        Ref<Internal::sVkDeferredRenderPassObject> mRenderPass = nullptr;
        Ref<Internal::sVkLightingRenderPassObject> mLightingRenderPass = nullptr;

        std::vector<Ref<Internal::sVkCommandBufferObject>> mCommandBufferObject   = {};
        Ref<Internal::sVkPipelineLayoutObject>             mCurrentPipelineLayout = nullptr;

        Ref<Buffer> mCurrentVertexBuffer = nullptr;
        Ref<Buffer> mCurrentIndexBuffer  = nullptr;
    };

} // namespace LTSE::Graphics