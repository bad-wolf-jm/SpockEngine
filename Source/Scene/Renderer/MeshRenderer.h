#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"

#include "Scene/VertexData.h"

#include "Graphics/Vulkan/VkAbstractRenderPass.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "SceneRenderPipeline.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct MeshRendererCreateInfo
    {
        bool     Opaque         = false;
        bool     IsTwoSided     = false;
        float    LineWidth      = 1.0f;
        fs::path VertexShader   = "";
        fs::path FragmentShader = "";

        Ref<sVkAbstractRenderPassObject> RenderPass = nullptr;

        bool operator==( const MeshRendererCreateInfo &p ) const
        {
            return ( IsTwoSided == p.IsTwoSided ) && ( LineWidth == p.LineWidth ) && ( VertexShader == p.VertexShader ) &&
                   ( FragmentShader == p.FragmentShader );
        }
    };

    struct MeshRendererCreateInfoHash
    {
        std::size_t operator()( const MeshRendererCreateInfo &node ) const
        {
            std::size_t h1 = std::hash<bool>()( node.IsTwoSided );
            std::size_t h2 = std::hash<float>()( node.LineWidth );
            std::size_t h3 = std::hash<std::string>()( node.VertexShader.string() );
            std::size_t h4 = std::hash<std::string>()( node.FragmentShader.string() );

            return h1 ^ h2 ^ h3 ^ h4;
        }
    };

    class MeshRenderer : public SceneRenderPipeline<VertexData>
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mMaterialID;
        };

        MeshRendererCreateInfo Spec = {};

        Ref<DescriptorSetLayout> CameraSetLayout  = nullptr;
        Ref<DescriptorSetLayout> NodeSetLayout    = nullptr;
        Ref<DescriptorSetLayout> TextureSetLayout = nullptr;

      public:
        MeshRenderer() = default;
        MeshRenderer( Ref<VkGraphicContext> aGraphicContext, MeshRendererCreateInfo const &aCreateInfo );

        static Ref<DescriptorSetLayout> GetCameraSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetTextureSetLayout( Ref<VkGraphicContext> aGraphicContext );
        static Ref<DescriptorSetLayout> GetNodeSetLayout( Ref<VkGraphicContext> aGraphicContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~MeshRenderer() = default;
    };

} // namespace SE::Core