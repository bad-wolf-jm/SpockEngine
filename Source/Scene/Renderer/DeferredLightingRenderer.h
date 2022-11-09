#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//DeferredLightingRenderContext.h"

#include "Scene/VertexData.h"

#include "Core/Vulkan/VkRenderPass.h"
#include "SceneRenderPipeline.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Graphics;
    namespace fs = std::filesystem;

    struct DeferredLightingRendererCreateInfo
    {
        bool Opaque = false;
        // bool     IsTwoSided     = false;
        // float    LineWidth      = 1.0f;
        // fs::path VertexShader   = "";
        // fs::path FragmentShader = "";

        Ref<LTSE::Graphics::Internal::sVkAbstractRenderPassObject> RenderPass = nullptr;

        // bool operator==( const DeferredLightingRendererCreateInfo &p ) const
        // {
        //     return ( IsTwoSided == p.IsTwoSided ) && ( LineWidth == p.LineWidth ) && ( VertexShader == p.VertexShader ) &&
        //            ( FragmentShader == p.FragmentShader );
        // }
    };

    // struct DeferredLightingRendererCreateInfoHash
    // {
    //     std::size_t operator()( const DeferredLightingRendererCreateInfo &node ) const
    //     {
    //         std::size_t h1 = std::hash<bool>()( node.IsTwoSided );
    //         std::size_t h2 = std::hash<float>()( node.LineWidth );
    //         std::size_t h3 = std::hash<std::string>()( node.VertexShader.string() );
    //         std::size_t h4 = std::hash<std::string>()( node.FragmentShader.string() );

    //         return h1 ^ h2 ^ h3 ^ h4;
    //     }
    // };

    class DeferredLightingRenderer : public SceneRenderPipeline<VertexData>
    {

      public:
        struct MaterialPushConstants
        {
            uint8_t mNumSamples;
        };

        DeferredLightingRendererCreateInfo Spec = {};

        Ref<DescriptorSetLayout> CameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> TextureSetLayout = nullptr;

      public:
        DeferredLightingRenderer() = default;
        DeferredLightingRenderer( GraphicContext &aGraphicContext, DeferredLightingRendererCreateInfo const &aCreateInfo );

        static Ref<DescriptorSetLayout> GetCameraSetLayout( GraphicContext &aGraphicContext );
        static Ref<DescriptorSetLayout> GetTextureSetLayout( GraphicContext &aGraphicContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~DeferredLightingRenderer() = default;
    };

} // namespace LTSE::Core