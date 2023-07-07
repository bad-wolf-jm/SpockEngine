#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/VertexData.h"

namespace SE::Core
{

    using namespace math;
    using namespace SE::Graphics;
    namespace fs = std::filesystem;

    struct MeshRendererCreateInfo
    {
        bool                Opaque         = false;
        bool                IsTwoSided     = false;
        float               LineWidth      = 1.0f;
        Ref<IShaderProgram> VertexShader   = nullptr;
        Ref<IShaderProgram> FragmentShader = nullptr;

        Ref<IRenderContext> RenderPass = nullptr;

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

    class MeshRenderer
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mMaterialID;
        };

        MeshRendererCreateInfo Spec = {};

        Ref<IDescriptorSetLayout> CameraSetLayout  = nullptr;
        Ref<IDescriptorSetLayout> NodeSetLayout    = nullptr;
        Ref<IDescriptorSetLayout> TextureSetLayout = nullptr;

      public:
        MeshRenderer() = default;
        MeshRenderer( Ref<IGraphicContext> aGraphicContext, MeshRendererCreateInfo const &aCreateInfo );

        static Ref<IDescriptorSetLayout> GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext );
        static Ref<IDescriptorSetLayout> GetNodeSetLayout( Ref<IGraphicContext> aGraphicContext );

        Ref<IGraphicsPipeline> Pipeline() { return mPipeline; }

        ~MeshRenderer() = default;

      private:
        Ref<IGraphicContext>   mGraphicContext = nullptr;
        Ref<IGraphicsPipeline> mPipeline       = nullptr;
    };

} // namespace SE::Core