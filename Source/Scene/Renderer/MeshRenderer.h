#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Scene/VertexData.h"

#include "Core/Vulkan/VkRenderPass.h"
#include "SceneRenderPipeline.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Graphics;
    namespace fs = std::filesystem;

    struct MeshRendererCreateInfo
    {
        bool     Opaque         = false;
        bool     IsTwoSided     = false;
        float    LineWidth      = 1.0f;
        fs::path VertexShader   = "";
        fs::path FragmentShader = "";

        Ref<LTSE::Graphics::Internal::sVkAbstractRenderPassObject> RenderPass = nullptr;

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
        MeshRenderer( GraphicContext &a_GraphicContext, MeshRendererCreateInfo const &a_CreateInfo );

        static Ref<DescriptorSetLayout> GetCameraSetLayout( GraphicContext &a_GraphicContext );
        static Ref<DescriptorSetLayout> GetTextureSetLayout( GraphicContext &a_GraphicContext );
        static Ref<DescriptorSetLayout> GetNodeSetLayout( GraphicContext &a_GraphicContext );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~MeshRenderer() = default;
    };

} // namespace LTSE::Core