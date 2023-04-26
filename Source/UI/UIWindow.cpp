#include "UIWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/Logging.h"

#include "Graphics/Vulkan/VkPipeline.h"

#include "Core/Profiling/BlockTimer.h"
#include "Graphics/Interface/IWindow.h"

#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"
#include "UIContext.h"

namespace SE::Core
{
    UIWindow::UIWindow( Ref<IGraphicContext> aGraphicContext, ImGuiViewport *aViewport )
        : mViewport{ aViewport }
        , mGraphicContext{ aGraphicContext }
    {
        mWindow        = SE::Core::New<IWindow>( (GLFWwindow *)aViewport->PlatformHandle );
        mSwapChain     = SE::Core::New<SwapChain>( GraphicContext<VkGraphicContext>(), mWindow );
        mRenderContext = SE::Graphics::ARenderContext( GraphicContext<VkGraphicContext>(), mSwapChain );

        CreatePipeline();

        mVertexBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = CreateBuffer( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    UIWindow::UIWindow( Ref<IGraphicContext> aGraphicContext, ARenderContext &aRenderContext )
        : mWindow{ nullptr }
        , mGraphicContext{ aGraphicContext }
        , mRenderContext{ aRenderContext }
        , mSwapChain{ nullptr }
        , mViewport{ nullptr }
    {
        CreatePipeline();

        mVertexBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = CreateBuffer( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    void UIWindow::CreatePipeline()
    {
        DescriptorBindingInfo lDescriptorBinding = { 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( GraphicContext<VkGraphicContext>(), lBindingLayout );

        std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        mUIVertexShader = New<ShaderModule>( GraphicContext<VkGraphicContext>(), lUIVertexShaderFiles, eShaderStageTypeFlags::VERTEX );
        std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        mUIFragmentShader =
            New<ShaderModule>( GraphicContext<VkGraphicContext>(), lUIFragmentShaderFiles, eShaderStageTypeFlags::FRAGMENT );
        GraphicsPipelineCreateInfo lUIPipelineCreateInfo = {};
        lUIPipelineCreateInfo.mShaderStages              = { { mUIVertexShader, "main" }, { mUIFragmentShader, "main" } };
        lUIPipelineCreateInfo.mInputBufferLayout          = {
            { "Position", eBufferDataType::VEC2, 0, 0 },
            { "TextureCoords", eBufferDataType::VEC2, 0, 1 },
            { "Color", eBufferDataType::COLOR, 0, 2 },
        };
        lUIPipelineCreateInfo.mTopology      = ePrimitiveTopology::TRIANGLES;
        lUIPipelineCreateInfo.mCulling       = eFaceCulling::NONE;
        lUIPipelineCreateInfo.mSampleCount   = mRenderContext.GetRenderTarget()->mSpec.mSampleCount;
        lUIPipelineCreateInfo.mLineWidth     = 1.0f;
        lUIPipelineCreateInfo.mRenderPass    = mRenderContext.GetRenderPass();
        lUIPipelineCreateInfo.mPushConstants = {
            { { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        };
        lUIPipelineCreateInfo.mSetLayouts = { mUIDescriptorSetLayout };

        mUIRenderPipeline = New<GraphicsPipeline>( GraphicContext<VkGraphicContext>(), lUIPipelineCreateInfo );
    }

    void UIWindow::SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        aRenderContext.Bind( mUIRenderPipeline );

        if( aDrawData->TotalVtxCount > 0 )
            aRenderContext.Bind( Cast<VkGpuBuffer>( mVertexBuffer ), Cast<VkGpuBuffer>( mIndexBuffer ) );

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        aRenderContext.GetCurrentCommandBuffer()->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );

        float lS[2];
        lS[0] = 2.0f / aDrawData->DisplaySize.x;
        lS[1] = 2.0f / aDrawData->DisplaySize.y;
        float lT[2];
        lT[0] = -1.0f - aDrawData->DisplayPos.x * lS[0];
        lT[1] = -1.0f - aDrawData->DisplayPos.y * lS[1];
        aRenderContext.PushConstants( { eShaderStageTypeFlags::VERTEX }, 0, lS );
        aRenderContext.PushConstants( { eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, lT );
    }

    // Render function
    void UIWindow::Render( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );
        if( ( lFramebufferWidth <= 0 ) || ( lFramebufferHeight <= 0 ) ) return;

        if( aDrawData->TotalVtxCount > 0 )
        {
            // Create or resize the vertex/index buffers
            size_t lVertexSize = aDrawData->TotalVtxCount * sizeof( ImDrawVert );
            size_t lIndexSize  = aDrawData->TotalIdxCount * sizeof( ImDrawIdx );

            mVertexBuffer->Resize( lVertexSize );
            mIndexBuffer->Resize( lIndexSize );

            int lVertexOffset = 0;
            int lIndexOffset  = 0;
            for( int n = 0; n < aDrawData->CmdListsCount; n++ )
            {
                const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
                mVertexBuffer->Upload( lImGuiDrawCommands->VtxBuffer.Data, lImGuiDrawCommands->VtxBuffer.Size, lVertexOffset );
                mIndexBuffer->Upload( lImGuiDrawCommands->IdxBuffer.Data, lImGuiDrawCommands->IdxBuffer.Size, lIndexOffset );

                lVertexOffset += lImGuiDrawCommands->VtxBuffer.Size;
                lIndexOffset += lImGuiDrawCommands->IdxBuffer.Size;
            }
        }

        // Setup desired Vulkan state
        SetupRenderState( aRenderContext, aDrawData );

        int lGlobalVtxOffset = 0;
        int lGlobalIdxOffset = 0;
        for( int n = 0; n < aDrawData->CmdListsCount; n++ )
        {
            const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];

            Render( aRenderContext, lImGuiDrawCommands, lGlobalVtxOffset, lGlobalIdxOffset, lFramebufferWidth, lFramebufferHeight,
                    aDrawData->DisplayPos, aDrawData->FramebufferScale );

            lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
            lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
        }
    }

    void UIWindow::Render( ARenderContext &aRenderContext, ImDrawList const *aDrawList, int aVertexOffset, int aIndexOffset,
                           int aFbWidth, int aFbHeight, ImVec2 aPosition, ImVec2 aScale )
    {
        ImVec4 lOffset = ImVec4{ aPosition.x, aPosition.y, aPosition.x, aPosition.y };
        ImVec4 lScale  = ImVec4{ aScale.x, aScale.y, aScale.x, aScale.y };

        for( int i = 0; i < aDrawList->CmdBuffer.Size; i++ )
        {
            const ImDrawCmd *lPcmd = &aDrawList->CmdBuffer[i];

            if( !( (VkDescriptorSet)lPcmd->TextureId ) ) continue;

            ImVec4 lClipRect = ( lPcmd->ClipRect - lOffset ) * lScale;
            if( !( ( lClipRect.x < aFbWidth ) && ( lClipRect.y < aFbHeight ) && ( lClipRect.z >= 0.0f ) && ( lClipRect.w >= 0.0f ) ) )
                continue;

            if( lClipRect.x < 0.0f ) lClipRect.x = 0.0f;
            if( lClipRect.y < 0.0f ) lClipRect.y = 0.0f;
            aRenderContext.GetCurrentCommandBuffer()->SetScissor(
                { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
                { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );

            VkDescriptorSet lDesc[1] = { (VkDescriptorSet)lPcmd->TextureId };
            vkCmdBindDescriptorSets( aRenderContext.GetCurrentCommandBuffer()->mVkObject, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     mUIRenderPipeline->GetVkPipelineLayoutObject()->mVkObject, 0, 1, lDesc, 0, NULL );

            aRenderContext.Draw( lPcmd->ElemCount, lPcmd->IdxOffset + aIndexOffset, lPcmd->VtxOffset + aVertexOffset, 1, 0 );
        }
    }

    void UIWindow::Render( ImDrawData *aDrawData )
    {
        mRenderContext.BeginRender();
        Render( mRenderContext, aDrawData );
    }

    void UIWindow::EndRender( ImDrawData *aDrawData )
    {
        mRenderContext.EndRender();
        mRenderContext.Present();
    }

    UIWindow::~UIWindow() {}

} // namespace SE::Core
