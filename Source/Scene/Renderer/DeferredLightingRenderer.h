#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"

// #include "Core/GraphicContext//DeferredLightingRenderContext.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"


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

        Ref<LTSE::Graphics::Internal::sVkAbstractRenderPassObject> RenderPass = nullptr;
    };

    class DeferredLightingRenderer : public SceneRenderPipeline<EmptyVertexData>
    {

      public:
        struct MaterialPushConstants
        {
            uint32_t mNumSamples;
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