#pragma once

#include "IGraphicBuffer.h"
#include "IGraphicContext.h"
#include "IGraphicsPipeline.h"
#include "IRenderTarget.h"
#include "IDescriptorSet.h"

namespace SE::Graphics
{
    class IRenderContext
    {
      public:
        IRenderContext() = default;
        IRenderContext( Ref<IGraphicContext> aGraphicContext, Ref<IRenderTarget> aRenderTarget );

        ~IRenderContext() = default;

        virtual bool BeginRender();
        virtual bool EndRender();
        virtual void Bind( Ref<IGraphicsPipeline> aGraphicPipeline );
        virtual void Bind( Ref<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 );
        virtual void Bind( Ref<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint = 0 );
        virtual void Bind( Ref<IGraphicBuffer> aVertexBuffer, Ref<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint = 0 );
        virtual void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize );

        template <typename _Ty>
        void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, const _Ty &aValue )
        {
            PushConstants( aShaderStages, aOffset, (void *)&aValue, sizeof( _Ty ) );
        }

        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0, uint32_t aInstanceCount = 0,
                   uint32_t aFirstInstance = 0 );

        virtual void InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

        virtual void InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 );

        void Present();
        void ResetBuffers();

      private:
        Ref<IGraphicContext>   mGraphicContext   = nullptr;
        Ref<IRenderTarget>     mRenderTarget     = nullptr;
        Ref<IGraphicsPipeline> mGraphicsPipeline = nullptr;
        Ref<IGraphicBuffer>    mVertexBuffer     = nullptr;
        Ref<IGraphicBuffer>    mIndexBuffer      = nullptr;

        bool mFrameIsStarted = false;
        bool mHasIndex       = false;
    };

} // namespace SE::Graphics