#include "IBLTools.h"

#include <chrono>

#include "../Primitives/Primitives.h"
#include "../VertexData.h"
#include "Developer/GraphicContext/ShaderModule.h"

#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Resource.h"

namespace LTSE::Core
{
    using namespace math::literals;
    using namespace LTSE::Core::Primitives;

    Ref<Texture2D> GenerateBRDFLUT( GraphicContext &a_GraphicContext, int32_t dim )
    {
        auto tStart = std::chrono::high_resolution_clock::now();

        TextureDescription l_BRDFLUTSpecification{};
        l_BRDFLUTSpecification.Sampled       = true;
        l_BRDFLUTSpecification.Memory        = { MemoryPropertyFlags::DEVICE_LOCAL };
        l_BRDFLUTSpecification.Usage         = { TextureUsageFlags::SAMPLED, TextureUsageFlags::COLOR_ATTACHMENT };
        l_BRDFLUTSpecification.MipLevels     = { { static_cast<uint32_t>( dim ), static_cast<uint32_t>( dim ), 1, 0 } };
        l_BRDFLUTSpecification.Format        = ColorFormat::RG32_FLOAT;
        Ref<Texture2D> l_BRDFLUT = a_GraphicContext.New<Texture2D>( l_BRDFLUTSpecification );

        std::vector<AttachmentDescription> l_Attachments = { AttachmentDescription{ 1,
                                                                                    AttachmentType::COLOR,
                                                                                    { 0.0f, 0.0f, 0.0f, 1.0f },
                                                                                    ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                                                                    ColorFormat::RG32_FLOAT,
                                                                                    AttachmentLoadOperation::CLEAR,
                                                                                    AttachmentStoreOperation::STORE,
                                                                                    AttachmentLoadOperation::DONT_CARE,
                                                                                    AttachmentStoreOperation::DONT_CARE,
                                                                                    ImageLayout::UNDEFINED,
                                                                                    ImageLayout::SHADER_READ_ONLY_OPTIMAL } };

        std::vector<SubpassDescription> l_Subpasses       = { { SubpassType::GRAPHICS, { 0 }, {}, -1 } };
        std::vector<DependencyDescription> l_Dependencies = {
            { { ExternalSubpass, { PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT }, { ResourceAccessFlags::COLOR_ATTACHMENT_WRITE, ResourceAccessFlags::COLOR_ATTACHMENT_READ } },
              { 0, { PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT }, { ResourceAccessFlags::COLOR_ATTACHMENT_WRITE, ResourceAccessFlags::COLOR_ATTACHMENT_READ } } } };
        RenderPassCreateInfo l_RenderPassCreateInfo = { 1, l_Attachments, l_Subpasses, l_Dependencies };
        Ref<RenderPass> l_RenderPass    = a_GraphicContext.New<RenderPass>( l_RenderPassCreateInfo );

        FramebufferCreateInfo l_FramebufferCI{};
        l_FramebufferCI.Width         = dim;
        l_FramebufferCI.Height        = dim;
        l_FramebufferCI.Attachments   = { l_BRDFLUT };
        l_FramebufferCI.RenderPass    = l_RenderPass;
        Framebuffer l_BRDFFramebuffer = Framebuffer( a_GraphicContext.GetDevice(), l_FramebufferCI );

        DescriptorSetLayoutCreateInfo l_TextureBindLayout{};
        l_TextureBindLayout.Bindings                          = {};
        Ref<DescriptorSetLayout> m_PipelineLayout = a_GraphicContext.New<DescriptorSetLayout>( l_TextureBindLayout );

        std::vector<std::string> l_VertexShaderFiles         = { GetResourcePath( "Shaders\\genbrdflut.vert.spv" ).string() };
        Ref<ShaderModule> l_VertexShaderModule   = a_GraphicContext.New<ShaderModule>( l_VertexShaderFiles, ShaderStageTypeFlags::VERTEX );
        std::vector<std::string> l_FragmentShaderFiles       = { GetResourcePath( "Shaders\\genbrdflut.frag.spv" ).string() };
        Ref<ShaderModule> l_FragmentShaderModule = a_GraphicContext.New<ShaderModule>( l_FragmentShaderFiles, ShaderStageTypeFlags::FRAGMENT );

        GraphicsPipelineCreateInfo m_BRDFRenderingPipelineCI      = {};
        m_BRDFRenderingPipelineCI.mShaderStages                    = { { l_VertexShaderModule, "main" }, { l_FragmentShaderModule, "main" } };
        m_BRDFRenderingPipelineCI.InputBufferLayout               = {};
        m_BRDFRenderingPipelineCI.Topology                        = PrimitiveTopology::TRIANGLES;
        m_BRDFRenderingPipelineCI.Culling                         = eFaceCulling::NONE;
        m_BRDFRenderingPipelineCI.Winding                         = FaceWinding::COUNTER_CLOCKWISE;
        m_BRDFRenderingPipelineCI.SampleCount                     = 1;
        m_BRDFRenderingPipelineCI.LineWidth                       = 1.0f;
        m_BRDFRenderingPipelineCI.RenderPass                      = l_RenderPass;
        m_BRDFRenderingPipelineCI.PushConstants                   = {};
        m_BRDFRenderingPipelineCI.SetLayouts                      = { m_PipelineLayout };
        Ref<GraphicsPipeline> l_BRDFRenderingPipeline = a_GraphicContext.New<GraphicsPipeline>( m_BRDFRenderingPipelineCI );

        auto l_CommandPool                                 = a_GraphicContext.New<CommandQueuePool>( a_GraphicContext.GetDevice()->GetGraphicsQueueFamily() );
        Ref<CommandQueue> l_BRDFRenderCommands = l_CommandPool->Allocate( 1 )[0];
        l_BRDFRenderCommands->BeginQueue();
        l_BRDFRenderCommands->BeginRenderPass( *l_RenderPass, l_BRDFFramebuffer, { dim, dim } );
        l_BRDFRenderCommands->SetViewport( { 0.0f, 0.0f }, { static_cast<float>( l_BRDFFramebuffer.Spec.Width ), static_cast<float>( l_BRDFFramebuffer.Spec.Height ) } );
        l_BRDFRenderCommands->SetScissor( { 0.0f, 0.0f }, { static_cast<float>( l_BRDFFramebuffer.Spec.Width ), static_cast<float>( l_BRDFFramebuffer.Spec.Height ) } );
        l_BRDFRenderCommands->Bind( l_BRDFRenderingPipeline );
        l_BRDFRenderCommands->Draw( 3, 0, 0, 1, 0 );
        l_BRDFRenderCommands->EndRenderPass();
        l_BRDFRenderCommands->EndQueue();
        l_BRDFRenderCommands->Submit();

        auto tEnd  = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>( tEnd - tStart ).count();
        // std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;

        return l_BRDFLUT;
    }

#define M_PI 3.14159265359

    Ref<TextureCubeMap> GenerateIBLCubemap( GraphicContext &a_GraphicContext, Ref<TextureCubeMap> a_Skybox, IBLCubemapType a_Type, int32_t dim )
    {
        auto tStart = std::chrono::high_resolution_clock::now();

        const uint32_t numMips = static_cast<uint32_t>( floor( log2( dim ) ) ) + 1;

        TextureDescription l_CubemapCI{};
        l_CubemapCI.Sampled   = true;
        l_CubemapCI.Memory    = { MemoryPropertyFlags::DEVICE_LOCAL };
        l_CubemapCI.Usage     = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_DESTINATION };
        l_CubemapCI.MipLevels = {};

        for( uint32_t l_Mip = 0; l_Mip < numMips; l_Mip++ )
        {
            float mip_dim = static_cast<float>( dim * std::pow( 0.5f, l_Mip ) );
            l_CubemapCI.MipLevels.push_back( { static_cast<uint32_t>( mip_dim ), static_cast<uint32_t>( mip_dim ), 1, 0 } );
        }

        l_CubemapCI.Format                        = ColorFormat::RGBA32_FLOAT;
        Ref<TextureCubeMap> l_CubeMap = a_GraphicContext.New<TextureCubeMap>( l_CubemapCI );

        std::vector<AttachmentDescription> l_Attachments = { AttachmentDescription{ 1,
                                                                                    AttachmentType::COLOR,
                                                                                    { 0.0f, 0.0f, 0.0f, 1.0f },
                                                                                    ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                                                                    ColorFormat::RGBA32_FLOAT,
                                                                                    AttachmentLoadOperation::CLEAR,
                                                                                    AttachmentStoreOperation::STORE,
                                                                                    AttachmentLoadOperation::DONT_CARE,
                                                                                    AttachmentStoreOperation::DONT_CARE,
                                                                                    ImageLayout::UNDEFINED,
                                                                                    ImageLayout::COLOR_ATTACHMENT_OPTIMAL } };

        std::vector<SubpassDescription> l_Subpasses       = { { SubpassType::GRAPHICS, { 0 }, {}, -1 } };
        std::vector<DependencyDescription> l_Dependencies = {
            { { ExternalSubpass, { PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT }, { ResourceAccessFlags::COLOR_ATTACHMENT_WRITE, ResourceAccessFlags::COLOR_ATTACHMENT_READ } },
              { 0, { PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT }, { ResourceAccessFlags::COLOR_ATTACHMENT_WRITE, ResourceAccessFlags::COLOR_ATTACHMENT_READ } } } };
        RenderPassCreateInfo l_RenderPassCreateInfo = { 1, l_Attachments, l_Subpasses, l_Dependencies };
        Ref<RenderPass> l_RenderPass    = a_GraphicContext.New<RenderPass>( l_RenderPassCreateInfo );

        TextureDescription l_OffscreenFramebufferCI{};
        l_OffscreenFramebufferCI.Sampled                  = true;
        l_OffscreenFramebufferCI.Memory                   = { MemoryPropertyFlags::DEVICE_LOCAL };
        l_OffscreenFramebufferCI.Usage                    = { TextureUsageFlags::SAMPLED, TextureUsageFlags::COLOR_ATTACHMENT, TextureUsageFlags::TRANSFER_SOURCE };
        l_OffscreenFramebufferCI.MipLevels                = { { static_cast<uint32_t>( dim ), static_cast<uint32_t>( dim ), 1, 0 } };
        l_OffscreenFramebufferCI.Format                   = ColorFormat::RGBA32_FLOAT;
        Ref<Texture2D> l_OffscreenFramebuffer = a_GraphicContext.New<Texture2D>( l_OffscreenFramebufferCI );

        FramebufferCreateInfo l_FramebufferCI{};
        l_FramebufferCI.Width       = dim;
        l_FramebufferCI.Height      = dim;
        l_FramebufferCI.Attachments = { l_OffscreenFramebuffer };
        l_FramebufferCI.RenderPass  = l_RenderPass;
        Framebuffer l_Framebuffer   = Framebuffer( a_GraphicContext.GetDevice(), l_FramebufferCI );

        // // Descriptors
        DescriptorSetLayoutCreateInfo l_TextureBindLayout{};
        l_TextureBindLayout.Bindings                          = { DescriptorBindingInfo{ 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { ShaderStageTypeFlags::FRAGMENT } } };
        Ref<DescriptorSetLayout> m_PipelineLayout = a_GraphicContext.New<DescriptorSetLayout>( l_TextureBindLayout );

        DescriptorPoolCreateInfo l_DescriptorPoolInfo = { 1000,
                                                          { { eDescriptorType::UNIFORM_BUFFER, 1000 },
                                                            { eDescriptorType::UNIFORM_BUFFER_DYNAMIC, 1000 },
                                                            { eDescriptorType::STORAGE_BUFFER, 1000 },
                                                            { eDescriptorType::COMBINED_IMAGE_SAMPLER, 1000 } } };
        auto l_DescriptorPool                         = a_GraphicContext.New<DescriptorPool>( l_DescriptorPoolInfo );

        Ref<DescriptorSet> l_PipelineDescriptors = l_DescriptorPool->Allocate( m_PipelineLayout );
        l_PipelineDescriptors->Write( a_Skybox, 0 );

        struct PushBlockIrradiance
        {
            math::mat4 mvp;
            float deltaPhi   = ( 2.0f * float( M_PI ) ) / 180.0f;
            float deltaTheta = ( 0.5f * float( M_PI ) ) / 64.0f;
        } pushBlockIrradiance;

        struct PushBlockPrefilterEnv
        {
            math::mat4 mvp;
            float roughness;
            uint32_t numSamples = 32u;
        } pushBlockPrefilterEnv;

        std::vector<std::string> l_SkyboxVertexShaderFiles       = { GetResourcePath( "Shaders\\filtercube.vert.spv" ).string() };
        Ref<ShaderModule> l_SkyboxVertexShaderModule = a_GraphicContext.New<ShaderModule>( l_SkyboxVertexShaderFiles, ShaderStageTypeFlags::VERTEX );

        std::vector<std::string> l_SkyboxFragmentShaderFiles;
        if( a_Type == PREFILTEREDENV )
            l_SkyboxFragmentShaderFiles = { GetResourcePath( "Shaders\\prefilterenvmap.frag.spv" ).string() };
        else
            l_SkyboxFragmentShaderFiles = { GetResourcePath( "Shaders\\irradiancecube.frag.spv" ).string() };

        Ref<ShaderModule> l_SkyboxFragmentShaderModule = a_GraphicContext.New<ShaderModule>( l_SkyboxFragmentShaderFiles, ShaderStageTypeFlags::FRAGMENT );

        GraphicsPipelineCreateInfo m_CubemapProcessorRenderingPipelineCI = {};
        m_CubemapProcessorRenderingPipelineCI.mShaderStages               = { { l_SkyboxVertexShaderModule, "main" }, { l_SkyboxFragmentShaderModule, "main" } };
        m_CubemapProcessorRenderingPipelineCI.InputBufferLayout          = VertexData::GetDefaultLayout();
        m_CubemapProcessorRenderingPipelineCI.Topology                   = PrimitiveTopology::TRIANGLES;
        m_CubemapProcessorRenderingPipelineCI.Culling                    = eFaceCulling::BACK;
        m_CubemapProcessorRenderingPipelineCI.Winding                    = FaceWinding::COUNTER_CLOCKWISE;
        m_CubemapProcessorRenderingPipelineCI.SampleCount                = 1;
        m_CubemapProcessorRenderingPipelineCI.LineWidth                  = 1.0f;
        m_CubemapProcessorRenderingPipelineCI.RenderPass                 = l_RenderPass;

        if( a_Type == PREFILTEREDENV )
            m_CubemapProcessorRenderingPipelineCI.PushConstants = { { { ShaderStageTypeFlags::VERTEX, ShaderStageTypeFlags::FRAGMENT }, 0, sizeof( PushBlockPrefilterEnv ) } };
        else
            m_CubemapProcessorRenderingPipelineCI.PushConstants = { { { ShaderStageTypeFlags::VERTEX, ShaderStageTypeFlags::FRAGMENT }, 0, sizeof( PushBlockIrradiance ) } };

        m_CubemapProcessorRenderingPipelineCI.SetLayouts = { m_PipelineLayout };

        Ref<GraphicsPipeline> l_CubemapRenderingPipeline = a_GraphicContext.New<GraphicsPipeline>( m_CubemapProcessorRenderingPipelineCI );

        std::vector<math::mat4> matrices = { math::Rotation( 180.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) ) * math::Rotation( 90.0_degf, math::vec3( 0.0f, 1.0f, 0.0f ) ),
                                             math::Rotation( 180.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) ) * math::Rotation( -90.0_degf, math::vec3( 0.0f, 1.0f, 0.0f ) ),
                                             math::Rotation( -90.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) ),
                                             math::Rotation( 90.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) ),
                                             math::Rotation( 180.0_degf, math::vec3( 1.0f, 0.0f, 0.0f ) ),
                                             math::Rotation( 180.0_degf, math::vec3( 0.0f, 0.0f, 1.0f ) ) };

        VertexBufferData l_Cube                = CreateCube();
        BufferDescription l_VBufferDescription = { { BufferBindTypeFlags::VERTEX_BUFFER }, { MemoryPropertyFlags::HOST_VISIBLE, MemoryPropertyFlags::HOST_COHERENT }, 0 };
        BufferDescription l_IBufferDescription = { { BufferBindTypeFlags::INDEX_BUFFER }, { MemoryPropertyFlags::HOST_VISIBLE, MemoryPropertyFlags::HOST_COHERENT }, 0 };
        Ref<Buffer> l_CubeVertices = Buffer::FromVector( a_GraphicContext.GetDevice(), l_VBufferDescription, l_Cube.Vertices );
        Ref<Buffer> l_CubeIndices  = Buffer::FromVector( a_GraphicContext.GetDevice(), l_IBufferDescription, l_Cube.Indices );

        l_CubeMap->TransitionImageLayout( VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );

        auto l_CommandPool                                 = a_GraphicContext.New<CommandQueuePool>( a_GraphicContext.GetDevice()->GetGraphicsQueueFamily() );
        Ref<CommandQueue> l_BRDFRenderCommands = l_CommandPool->Allocate( 1 )[0];
        l_BRDFRenderCommands->BeginQueue();

        for( uint32_t l_MipLevel = 0; l_MipLevel < numMips; l_MipLevel++ )
        {
            for( uint32_t l_CubeFace = 0; l_CubeFace < 6; l_CubeFace++ )
            {
                l_BRDFRenderCommands->BeginRenderPass( *l_RenderPass, l_Framebuffer, { dim, dim } );
                l_BRDFRenderCommands->SetViewport(
                    { 0.0f, 0.0f }, { static_cast<float>( l_CubeMap->Spec.MipLevels[l_MipLevel].Width ), static_cast<float>( l_CubeMap->Spec.MipLevels[l_MipLevel].Height ) } );
                l_BRDFRenderCommands->SetScissor(
                    { 0.0f, 0.0f }, { static_cast<float>( l_CubeMap->Spec.MipLevels[l_MipLevel].Width ), static_cast<float>( l_CubeMap->Spec.MipLevels[l_MipLevel].Height ) } );
                l_BRDFRenderCommands->Bind( l_CubemapRenderingPipeline );
                l_BRDFRenderCommands->Bind( l_PipelineDescriptors, 0, -1 );

                switch( a_Type )
                {
                case IRRADIANCE:
                    pushBlockIrradiance.mvp = math::Perspective( (float)( M_PI / 2.0 ), 1.0f, 0.1f, 512.0f ) * matrices[l_CubeFace];
                    l_BRDFRenderCommands->PushConstants( { ShaderStageTypeFlags::VERTEX, ShaderStageTypeFlags::FRAGMENT }, 0, pushBlockIrradiance );
                    break;
                case PREFILTEREDENV:
                    pushBlockPrefilterEnv.mvp       = math::Perspective( (float)( M_PI / 2.0 ), 1.0f, 0.1f, 512.0f ) * matrices[l_CubeFace];
                    pushBlockPrefilterEnv.roughness = (float)l_MipLevel / (float)( numMips - 1 );
                    l_BRDFRenderCommands->PushConstants( { ShaderStageTypeFlags::VERTEX, ShaderStageTypeFlags::FRAGMENT }, 0, pushBlockPrefilterEnv );
                    break;
                };

                l_BRDFRenderCommands->Bind( l_CubeVertices, l_CubeIndices );
                l_BRDFRenderCommands->Draw( l_Cube.Indices.size(), 0, 0, 1, 0 );

                l_BRDFRenderCommands->EndRenderPass();

                float D = static_cast<float>( dim * std::pow( 0.5f, l_MipLevel ) );
                l_CubeMap->CopyImage( l_BRDFRenderCommands->GetVkCommandBufferObject(), l_OffscreenFramebuffer, { 0.0f, 0.0f, D, D }, l_CubeFace, l_MipLevel );
            }
        }
        l_BRDFRenderCommands->EndQueue();
        l_BRDFRenderCommands->Submit();

        l_CubeMap->TransitionImageLayout( VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        auto tEnd  = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>( tEnd - tStart ).count();
        // std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;

        return l_CubeMap;
    }

} // namespace LTSE::Core
