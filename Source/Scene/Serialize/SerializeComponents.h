
#include <string>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Scene/Components.h"
#include "FileIO.h"

namespace SE::Core
{
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core::EntityComponentSystem::Components;

    using EntityMap = std::unordered_map<std::string, Entity>;
    using BufferMap = std::unordered_map<std::string, Ref<VkGpuBuffer>>;

    struct sReadContext
    {
        EntityMap mEntities;
        BufferMap mBuffers;
    };

    void ReadComponent( sTag &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sCameraComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sActorComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAnimationChooser &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAnimationComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext,
                        std::vector<sImportedAnimationSampler> &aInterpolationData );
    void ReadComponent( sAnimatedTransformComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sNodeTransformComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sTransformMatrixComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sStaticMeshComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sParticleSystemComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sParticleShaderComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sWireframeComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sWireframeMeshComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sBoundingBoxComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sSkeletonComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sRayTracingTargetComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sMaterialComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sMaterialShaderComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sBackgroundComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAmbientLightingComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );
    void ReadComponent( sLightComponent &aComponent, ConfigurationNode const &aNode, sReadContext &aReadConext );

    void DoWriteComponent( ConfigurationWriter &aOut, sTag const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sRelationshipComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sCameraComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sAnimationChooser const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sActorComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sAnimatedTransformComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sNodeTransformComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sTransformMatrixComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sStaticMeshComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sParticleSystemComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sParticleShaderComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sSkeletonComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sWireframeComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sWireframeMeshComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sBoundingBoxComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sRayTracingTargetComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sMaterialComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sMaterialShaderComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sBackgroundComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sAmbientLightingComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, sLightComponent const &aComponent );
} // namespace SE::Core