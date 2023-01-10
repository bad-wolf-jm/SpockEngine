
#include <string>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "FileIO.h"
#include "Scene/Components.h"

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

    std::string const &GetTypeTag( std::string const &aTypeName );

    template <typename _Ty>
    bool HasTypeTag( ConfigurationNode const &aNode )
    {
        auto lInternalTypeName = std::string( typeid( _Ty ).name() );
        return ( !aNode[GetTypeTag( lInternalTypeName )].IsNull() );
    }

    template <typename _Ty>
    std::string TypeTag()
    {
        auto lInternalTypeName = std::string( typeid( _Ty ).name() );
        return ( GetTypeTag( lInternalTypeName ) );
    }

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

    void WriteComponent( ConfigurationWriter &aOut, sTag const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sRelationshipComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sCameraComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAnimationChooser const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sActorComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAnimatedTransformComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sNodeTransformComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sTransformMatrixComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sStaticMeshComponent const &aComponent, std::string const &aMeshPath );
    void WriteComponent( ConfigurationWriter &aOut, sParticleSystemComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sParticleShaderComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sSkeletonComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sWireframeComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sWireframeMeshComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sBoundingBoxComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sRayTracingTargetComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sMaterialComponent const &aComponent, std::string const &aMaterialPath );
    void WriteComponent( ConfigurationWriter &aOut, sMaterialShaderComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sBackgroundComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAmbientLightingComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sLightComponent const &aComponent );
} // namespace SE::Core