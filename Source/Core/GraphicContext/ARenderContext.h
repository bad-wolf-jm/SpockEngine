#pragma once

#include "Core/GraphicContext//GraphicContext.h"

#include "ARenderTarget.h"
#include "GraphicsPipeline.h"

namespace LTSE::Graphics
{
    using namespace Internal;

    class ARenderContext
    {
      public:
        ARenderContext() = default;
        ARenderContext( GraphicContext const &aGraphicContext, Ref<ARenderTarget> aRenderTarget );

        ~ARenderContext() = default;

        GraphicContext &GetGraphicContext() { return mGraphicContext; };

        uint32_t                         GetOutputImageCount();
        Ref<ARenderTarget>               GetRenderTarget() { return mRenderTarget; }
        Ref<sVkCommandBufferObject>      GetCurrentCommandBuffer() { return mRenderTarget->GetCommandBuffer( mCurrentCommandBuffer ); }
        Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderTarget->GetRenderPass(); }

        bool BeginRender();
        bool EndRender();

        void Bind( Ref<GraphicsPipeline> aGraphicPipeline );
        void Bind( Ref<DescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 );
        void Bind( Ref<Buffer> aVertexBuffer, uint32_t aBindPoint = 0 );
        void Bind( Ref<Buffer> aVertexBuffer, Ref<Buffer> aIndexBuffer, uint32_t aBindPoint = 0 );

        template <typename T>
        void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, const T &aValue )
        {
            auto lCommandBuffer = mRenderTarget->GetCommandBuffer( mCurrentCommandBuffer );
            lCommandBuffer->PushConstants<T>( (VkShaderStageFlags)aShaderStages, aOffset, aValue, mCurrentPipelineLayout );
        }

        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0, uint32_t aInstanceCount = 0,
            uint32_t aFirstInstance = 0 );

        void Present();

        void ResetBuffers();

        operator bool() { return static_cast<bool>( mRenderTarget ) && static_cast<bool>( mRenderTarget->GetRenderPass() ); }

      private:
        GraphicContext mGraphicContext{};

        bool mFrameIsStarted = false;

        uint32_t mCurrentCommandBuffer = 0;

        Ref<ARenderTarget> mRenderTarget = nullptr;

        std::vector<Ref<sVkCommandBufferObject>> mCommandBufferObject   = {};
        Ref<sVkPipelineLayoutObject>             mCurrentPipelineLayout = nullptr;

        Ref<Buffer> mCurrentVertexBuffer = nullptr;
        Ref<Buffer> mCurrentIndexBuffer  = nullptr;
    };

} // namespace LTSE::Graphics