#include "UIContext.h"

#include "Core/Logging.h"
#include "Core/Textures/ColorFormat.h"

#include "Core/Vulkan/VkPipeline.h"

#include "Core/Platform/Window.h"
#include "Core/Profiling/BlockTimer.h"

// std
#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"

namespace LTSE::Core
{

    void UIContext::SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        LTSE_PROFILE_FUNCTION();

        aRenderContext.Bind( mUIRenderPipeline );

        if( aDrawData->TotalVtxCount > 0 ) aRenderContext.Bind( mVertexBuffer, mIndexBuffer );

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        aRenderContext.GetCurrentCommandBuffer()->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );

        // Setup scale and translation:
        // Our visible imgui space lies from aDrawData->DisplayPps (top left) to aDrawData->DisplayPos+data_data->DisplaySize (bottom
        // right). DisplayPos is (0,0) for single viewport apps.
        {
            float lScale[2];
            lScale[0] = 2.0f / aDrawData->DisplaySize.x;
            lScale[1] = 2.0f / aDrawData->DisplaySize.y;
            float translate[2];
            translate[0] = -1.0f - aDrawData->DisplayPos.x * lScale[0];
            translate[1] = -1.0f - aDrawData->DisplayPos.y * lScale[1];
            aRenderContext.PushConstants( { Graphics::Internal::eShaderStageTypeFlags::VERTEX }, 0, lScale );
            aRenderContext.PushConstants( { Graphics::Internal::eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, translate );
        }
    }

    // Render function
    void UIContext::RenderDrawData( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        LTSE_PROFILE_FUNCTION();

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
                lVertexOffset += lImGuiDrawCommands->VtxBuffer.Size * sizeof( ImDrawVert );
                lIndexOffset += lImGuiDrawCommands->IdxBuffer.Size * sizeof( ImDrawIdx );
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

                    aRenderContext.GetCurrentCommandBuffer()->SetScissor( { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
                        { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );

                    // Bind a the descriptor set for the current texture.
                    if( (VkDescriptorSet)lPcmd->TextureId )
                    {
                        VkDescriptorSet desc_set[1] = { (VkDescriptorSet)lPcmd->TextureId };
                        vkCmdBindDescriptorSets( aRenderContext.GetCurrentCommandBuffer()->mVkObject, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mUIRenderPipeline->GetVkPipelineLayoutObject()->mVkObject, 0, 1, desc_set, 0, NULL );

                        aRenderContext.Draw(
                            lPcmd->ElemCount, lPcmd->IdxOffset + lGlobalIdxOffset, lPcmd->VtxOffset + lGlobalVtxOffset, 1, 0 );
                    }
                }
            }
            lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
            lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
        }
    }

    UIContext::UIContext( Ref<LTSE::Core::ViewportClient> aWindow, GraphicContext &aGraphicContext, ARenderContext &aRenderContext,
        std::string &aImGuiConfigPath )
        : mGraphicContext{ aGraphicContext }
        , mImGuiConfigPath{ aImGuiConfigPath }
    {
        IMGUI_CHECKVERSION();
        mImGUIOverlay  = ImGui::CreateContext();
        mImPlotContext = ImPlot::CreateContext();

        ImGui::SetCurrentContext( mImGUIOverlay );
        ImGuiIO &io            = ImGui::GetIO();
        io.BackendRendererName = "imgui_impl_vulkan";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
        if( !mImGuiConfigPath.empty() )
            io.IniFilename = mImGuiConfigPath.c_str();
        else
            io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan( aWindow->GetGLFWWindow(), true );

        DescriptorBindingInfo lDescriptorBinding = {
            0, Internal::eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        mUIVertexShader                  = New<Graphics::Internal::ShaderModule>( mGraphicContext.mContext, lUIVertexShaderFiles,
                                                                 Graphics::Internal::eShaderStageTypeFlags::VERTEX );

        std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        mUIFragmentShader                  = New<Graphics::Internal::ShaderModule>( mGraphicContext.mContext, lUIFragmentShaderFiles,
                                                                   Graphics::Internal::eShaderStageTypeFlags::FRAGMENT );
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
            { { Graphics::Internal::eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        };
        lUIPipelineCreateInfo.SetLayouts = { mUIDescriptorSetLayout };

        mUIRenderPipeline               = New<GraphicsPipeline>( mGraphicContext, lUIPipelineCreateInfo );
        ImGuiViewport *lMainViewport    = ImGui::GetMainViewport();
        lMainViewport->RendererUserData = nullptr;

        static ImWchar lRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        ImFontConfig   lFontConfig;
        lFontConfig.MergeMode        = true;
        lFontConfig.PixelSnapH       = true;
        lFontConfig.GlyphMinAdvanceX = 10.0f;

        const float lFontSize = 14.0f;

        mMonoFont = io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSansMono.ttf" ).string().c_str(), lFontSize );
        io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\fontawesome-webfont.ttf" ).string().c_str(), lFontSize, &lFontConfig, lRanges );

        mMainFont = io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSans.ttf" ).string().c_str(), lFontSize );
        io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\fontawesome-webfont.ttf" ).string().c_str(), lFontSize, &lFontConfig, lRanges );

        mBoldFont = io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSans-Bold.ttf" ).string().c_str(), lFontSize );
        io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\fontawesome-webfont.ttf" ).string().c_str(), lFontSize, &lFontConfig, lRanges );

        mObliqueFont = io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSans-Oblique.ttf" ).string().c_str(), lFontSize );
        io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\fontawesome-webfont.ttf" ).string().c_str(), lFontSize, &lFontConfig, lRanges );

        mBoldObliqueFont = io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSans-BoldOblique.ttf" ).string().c_str(), lFontSize );
        io.Fonts->AddFontFromFileTTF(
            GetResourcePath( "fonts\\fontawesome-webfont.ttf" ).string().c_str(), lFontSize, &lFontConfig, lRanges );

        mUIStyle = UIStyle{ true };

        unsigned char *lFontPixelData;
        int            lWidth, lHeight;
        io.Fonts->GetTexDataAsRGBA32( &lFontPixelData, &lWidth, &lHeight );
        size_t lUploadSize = lWidth * lHeight * 4 * sizeof( char );

        TextureDescription lTextureDesc;
        lTextureDesc.IsHostVisible       = false;
        lTextureDesc.Usage               = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_DESTINATION };
        lTextureDesc.MinificationFilter  = SamplerFilter::LINEAR;
        lTextureDesc.MagnificationFilter = SamplerFilter::LINEAR;
        lTextureDesc.MipmapMode          = SamplerMipmap::LINEAR;
        lTextureDesc.WrappingMode        = SamplerWrapping::REPEAT;
        lTextureDesc.MipLevels           = { { static_cast<uint32_t>( lWidth ), static_cast<uint32_t>( lHeight ), 0, 0 } };
        lTextureDesc.Format              = eColorFormat::RGBA8_UNORM;
        lTextureDesc.Sampled             = true;

        LTSE::Graphics::TextureData lTextureData;
        lTextureData.ByteSize = lUploadSize;
        lTextureData.Data     = lFontPixelData;

        mFontTexture       = New<Texture2D>( mGraphicContext, lTextureDesc, lTextureData );
        mFontDescriptorSet = AddTexture( mFontTexture );
        io.Fonts->TexID    = (ImTextureID)mFontDescriptorSet->GetVkDescriptorSet();

        mVertexBuffer = New<Buffer>( mGraphicContext, eBufferBindType::VERTEX_BUFFER, true, false, true, true, 1 );
        mIndexBuffer  = New<Buffer>( mGraphicContext, eBufferBindType::INDEX_BUFFER, true, false, true, true, 1 );
    }

    UIContext::~UIContext()
    {
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext( mImPlotContext );
        ImGui::DestroyContext( mImGUIOverlay );
    }

    void UIContext::BeginFrame()
    {
        ImGui::SetCurrentContext( mImGUIOverlay );
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
        ImGuizmo::AllowAxisFlip( false );
        ImGui::PushFont( mMainFont );
    }

    void UIContext::EndFrame( ARenderContext &aRenderContext )
    {
        ImGui::PopFont();
        ImGui::Render();
        ImDrawData *drawdata = ImGui::GetDrawData();
        RenderDrawData( aRenderContext, drawdata );
    }

    ImageHandle UIContext::CreateTextureHandle( Ref<Texture2D> aTexture ) { return ImageHandle{ AddTexture( aTexture ) }; }

    ImGuiIO &UIContext::GetIO()
    {
        ImGui::SetCurrentContext( mImGUIOverlay );
        return ImGui::GetIO();
    }

    void UIContext::PushFontFamily( FontFamily aFamily )
    {
        if( ( aFamily & FontFamilyFlags::BOLD ) && ( aFamily & FontFamilyFlags::ITALIC ) )
            ImGui::PushFont( mBoldObliqueFont );
        else if( ( aFamily & FontFamilyFlags::BOLD ) )
            ImGui::PushFont( mBoldFont );
        else if( ( aFamily & FontFamilyFlags::ITALIC ) )
            ImGui::PushFont( mObliqueFont );
        else if( ( aFamily & FontFamilyFlags::MONO ) )
            ImGui::PushFont( mMonoFont );
        else
            ImGui::PushFont( mMainFont );
    }

    void UIContext::PopFont() { ImGui::PopFont(); }

    Ref<DescriptorSet> UIContext::AddTexture( Ref<Texture2D> aTexture )
    {
        Ref<DescriptorSet> lDescriptorSet = New<DescriptorSet>( mGraphicContext, mUIDescriptorSetLayout );
        lDescriptorSet->Write( aTexture, 0 );
        return lDescriptorSet;
    }

} // namespace LTSE::Core
