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
        IRenderContext( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderTarget> aRenderTarget );

        ~IRenderContext() = default;

        virtual bool BeginRender();
        virtual bool EndRender();
        virtual void Bind( ref_t<IGraphicsPipeline> aGraphicPipeline );
        virtual void Bind( void* aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 ) = 0;
        virtual void Bind( ref_t<IDescriptorSet> aDescriptorSet, uint32_t aSetIndex, int32_t aDynamicOffset = -1 ) = 0;
        virtual void Bind( ref_t<IGraphicBuffer> aVertexBuffer, uint32_t aBindPoint = 0 );
        virtual void Bind( ref_t<IGraphicBuffer> aVertexBuffer, ref_t<IGraphicBuffer> aIndexBuffer, uint32_t aBindPoint = 0 );
        virtual void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, void *aValue, uint32_t aSize ) = 0;
        virtual void SetViewport( math::ivec2 aOffset, math::uvec2 aSize ) = 0;
        virtual void SetScissor( math::ivec2 aOffset, math::uvec2 aSize ) = 0;

        template <typename _Ty>
        void PushConstants( ShaderStageType aShaderStages, uint32_t aOffset, const _Ty &aValue )
        {
            PushConstants( aShaderStages, aOffset, (void *)&aValue, sizeof( _Ty ) );
        }

        void Draw( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0, uint32_t aInstanceCount = 0,
                   uint32_t aFirstInstance = 0 );

        virtual void InternalDrawIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 ) = 0;

        virtual void InternalDrawNonIndexed( uint32_t aVertexCount, uint32_t aVertexOffset, uint32_t aVertexBufferOffset = 0,
                                     uint32_t aInstanceCount = 0, uint32_t aFirstInstance = 0 ) = 0;

        void Present();
        void ResetBuffers();

      protected:
        ref_t<IGraphicContext>   mGraphicContext   = nullptr;
        ref_t<IRenderTarget>     mRenderTarget     = nullptr;
        ref_t<IGraphicsPipeline> mGraphicsPipeline = nullptr;
        ref_t<IGraphicBuffer>    mVertexBuffer     = nullptr;
        ref_t<IGraphicBuffer>    mIndexBuffer      = nullptr;

        bool mFrameIsStarted = false;
        bool mHasIndex       = false;
    };

} // namespace SE::Graphics