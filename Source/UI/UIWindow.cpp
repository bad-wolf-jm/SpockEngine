#include "UIWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/Logging.h"

#include "Graphics/Vulkan/VkPipeline.h"

#include "Core/Profiling/BlockTimer.h"
#include "Graphics/Interface/IWindow.h"

#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"

namespace SE::Core
{
    UIWindow::UIWindow( Ref<VkGraphicContext> aGraphicContext, ImGuiViewport *aViewport )
        : mViewport{ aViewport }
        , mGraphicContext{ aGraphicContext }
    {
        mWindow        = SE::Core::New<IWindow>( (GLFWwindow *)aViewport->PlatformHandle );
        mSwapChain     = SE::Core::New<SwapChain>( mGraphicContext, mWindow );
        mRenderContext = SE::Graphics::ARenderContext( mGraphicContext, mSwapChain );

        DescriptorBindingInfo lDescriptorBinding = {
            0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        mUIVertexShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIVertexShaderFiles, Graphics::eShaderStageTypeFlags::VERTEX );

        std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        mUIFragmentShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIFragmentShaderFiles, Graphics::eShaderStageTypeFlags::FRAGMENT );
        GraphicsPipelineCreateInfo lUIPipelineCreateInfo = {};
        lUIPipelineCreateInfo.mShaderStages              = { { mUIVertexShader, "main" }, { mUIFragmentShader, "main" } };
        lUIPipelineCreateInfo.InputBufferLayout          = {
            { "Position", eBufferDataType::VEC2, 0, 0 },
            { "TextureCoords", eBufferDataType::VEC2, 0, 1 },
            { "Color", eBufferDataType::COLOR, 0, 2 },
        };
        lUIPipelineCreateInfo.Topology      = ePrimitiveTopology::TRIANGLES;
        lUIPipelineCreateInfo.Culling       = eFaceCulling::NONE;
        lUIPipelineCreateInfo.SampleCount   = mRenderContext.GetRenderTarget()->mSpec.mSampleCount;
        lUIPipelineCreateInfo.LineWidth     = 1.0f;
        lUIPipelineCreateInfo.RenderPass    = mRenderContext.GetRenderPass();
        lUIPipelineCreateInfo.PushConstants = {
            { { Graphics::eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        };
        lUIPipelineCreateInfo.SetLayouts = { mUIDescriptorSetLayout };

        mUIRenderPipeline = New<GraphicsPipeline>( mGraphicContext, lUIPipelineCreateInfo );
        mVertexBuffer     = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer      = New<VkGpuBuffer>( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    UIWindow::UIWindow( Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext )
        : mWindow{ nullptr }
        , mGraphicContext{ aGraphicContext }
        , mSwapChain{ nullptr }
        , mViewport{ nullptr }
    {

        DescriptorBindingInfo lDescriptorBinding = {
            0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        mUIVertexShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIVertexShaderFiles, Graphics::eShaderStageTypeFlags::VERTEX );

        std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        mUIFragmentShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIFragmentShaderFiles, Graphics::eShaderStageTypeFlags::FRAGMENT );
        GraphicsPipelineCreateInfo lUIPipelineCreateInfo = {};
        lUIPipelineCreateInfo.mShaderStages              = { { mUIVertexShader, "main" }, { mUIFragmentShader, "main" } };
        lUIPipelineCreateInfo.InputBufferLayout          = {
            { "Position", eBufferDataType::VEC2, 0, 0 },
            { "TextureCoords", eBufferDataType::VEC2, 0, 1 },
            { "Color", eBufferDataType::COLOR, 0, 2 },
        };
        lUIPipelineCreateInfo.Topology      = ePrimitiveTopology::TRIANGLES;
        lUIPipelineCreateInfo.Culling       = eFaceCulling::NONE;
        lUIPipelineCreateInfo.SampleCount   = aRenderContext.GetRenderTarget()->mSpec.mSampleCount;
        lUIPipelineCreateInfo.LineWidth     = 1.0f;
        lUIPipelineCreateInfo.RenderPass    = aRenderContext.GetRenderPass();
        lUIPipelineCreateInfo.PushConstants = {
            { { Graphics::eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        };
        lUIPipelineCreateInfo.SetLayouts = { mUIDescriptorSetLayout };

        mUIRenderPipeline = New<GraphicsPipeline>( mGraphicContext, lUIPipelineCreateInfo );
        mVertexBuffer     = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer      = New<VkGpuBuffer>( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    void UIWindow::SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        aRenderContext.Bind( mUIRenderPipeline );

        if( aDrawData->TotalVtxCount > 0 ) aRenderContext.Bind( mVertexBuffer, mIndexBuffer );

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        aRenderContext.GetCurrentCommandBuffer()->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );

        float lScale[2];
        lScale[0] = 2.0f / aDrawData->DisplaySize.x;
        lScale[1] = 2.0f / aDrawData->DisplaySize.y;
        float translate[2];
        translate[0] = -1.0f - aDrawData->DisplayPos.x * lScale[0];
        translate[1] = -1.0f - aDrawData->DisplayPos.y * lScale[1];
        aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, 0, lScale );
        aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, translate );
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
            size_t vertex_size = aDrawData->TotalVtxCount * sizeof( ImDrawVert );
            size_t index_size  = aDrawData->TotalIdxCount * sizeof( ImDrawIdx );

            mVertexBuffer->Resize( vertex_size );
            mIndexBuffer->Resize( index_size );

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

        // Will project scissor/clipping rectangles into framebuffer space
        ImVec2 lClipOffset = aDrawData->DisplayPos;       // (0,0) unless using multi-viewports
        ImVec2 lClipScale  = aDrawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

        int lGlobalVtxOffset = 0;
        int lGlobalIdxOffset = 0;
        for( int n = 0; n < aDrawData->CmdListsCount; n++ )
        {
            const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
            for( int cmd_i = 0; cmd_i < lImGuiDrawCommands->CmdBuffer.Size; cmd_i++ )
            {
                const ImDrawCmd *lPcmd = &lImGuiDrawCommands->CmdBuffer[cmd_i];

                // Project scissor/clipping rectangles into framebuffer space
                ImVec4 lClipRect;
                lClipRect.x = ( lPcmd->ClipRect.x - lClipOffset.x ) * lClipScale.x;
                lClipRect.y = ( lPcmd->ClipRect.y - lClipOffset.y ) * lClipScale.y;
                lClipRect.z = ( lPcmd->ClipRect.z - lClipOffset.x ) * lClipScale.x;
                lClipRect.w = ( lPcmd->ClipRect.w - lClipOffset.y ) * lClipScale.y;

                if( lClipRect.x < lFramebufferWidth && lClipRect.y < lFramebufferHeight && lClipRect.z >= 0.0f && lClipRect.w >= 0.0f )
                {
                    // Negative offsets are illegal for vkCmdSetScissor
                    if( lClipRect.x < 0.0f ) lClipRect.x = 0.0f;
                    if( lClipRect.y < 0.0f ) lClipRect.y = 0.0f;

                    aRenderContext.GetCurrentCommandBuffer()->SetScissor(
                        { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
                        { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );

                    // Bind a the descriptor set for the current texture.
                    if( (VkDescriptorSet)lPcmd->TextureId )
                    {
                        VkDescriptorSet desc_set[1] = { (VkDescriptorSet)lPcmd->TextureId };
                        vkCmdBindDescriptorSets( aRenderContext.GetCurrentCommandBuffer()->mVkObject, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                 mUIRenderPipeline->GetVkPipelineLayoutObject()->mVkObject, 0, 1, desc_set, 0, NULL );

                        aRenderContext.Draw( lPcmd->ElemCount, lPcmd->IdxOffset + lGlobalIdxOffset,
                                             lPcmd->VtxOffset + lGlobalVtxOffset, 1, 0 );
                    }
                }
            }
            lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
            lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
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

    UIWindow::~UIWindow()
    {
    }

} // namespace SE::Core
