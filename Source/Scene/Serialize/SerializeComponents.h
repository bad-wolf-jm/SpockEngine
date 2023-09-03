
#include <string>
#include <unordered_map>

#include "Core/Entity/Collection.h"

#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkGraphicsPipeline.h"

#include "FileIO.h"
#include "Scene/Components.h"

namespace SE::Core
{
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core::EntityComponentSystem::Components;

    using EntityMap = std::unordered_map<string_t, Entity>;
    using BufferMap = std::unordered_map<string_t, ref_t<VkGpuBuffer>>;

    struct sReadContext
    {
        EntityMap mEntities;
        BufferMap mBuffers;
    };

    string_t const &GetTypeTag( string_t const &aTypeName );

    template <typename _Ty>
    bool HasTypeTag( YAML::Node const &aNode )
    {
        auto lInternalTypeName = string_t( typeid( _Ty ).name() );
        return static_cast<bool>( aNode[GetTypeTag( lInternalTypeName )] );
    }

    template <typename _Ty>
    string_t TypeTag()
    {
        auto lInternalTypeName = string_t( typeid( _Ty ).name() );
        return ( GetTypeTag( lInternalTypeName ) );
    }

    math::mat4 ReadMatrix( YAML::Node const &aNode );

    template <typename _Ty>
    _Ty Get( YAML::Node const &aNode, _Ty aDefault )
    {
        if( ( !aNode ) || ( aNode.IsNull() ) )
            return aDefault;

        return aNode.as<_Ty>();
    }

    math::vec2 Get( YAML::Node const &aNode, std::array<string_t, 2> const &aKeys, math::vec2 const &aDefault );
    math::vec3 Get( YAML::Node const &aNode, std::array<string_t, 3> const &aKeys, math::vec3 const &aDefault );
    math::vec4 Get( YAML::Node const &aNode, std::array<string_t, 4> const &aKeys, math::vec4 const &aDefault );

    void ReadComponent( sTag &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sCameraComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sActorComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAnimationChooser &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAnimationComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext,
                        vec_t<sImportedAnimationSampler> &aInterpolationData );
    void ReadComponent( sAnimatedTransformComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sNodeTransformComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sStaticMeshComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    // void ReadComponent( sParticleSystemComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    // void ReadComponent( sParticleShaderComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sWireframeComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sWireframeMeshComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sBoundingBoxComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sSkeletonComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sRayTracingTargetComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sMaterialComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    // void ReadComponent( sMaterialShaderComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sBackgroundComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sAmbientLightingComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sLightComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );
    void ReadComponent( sUIComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext );

    void WriteComponent( ConfigurationWriter &aOut, sTag const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sRelationshipComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sCameraComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAnimationChooser const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sActorComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAnimatedTransformComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sNodeTransformComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sStaticMeshComponent const &aComponent, string_t const &aMeshPath );
    // void WriteComponent( ConfigurationWriter &aOut, sParticleSystemComponent const &aComponent );
    // void WriteComponent( ConfigurationWriter &aOut, sParticleShaderComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sSkeletonComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sWireframeComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sWireframeMeshComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sBoundingBoxComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sRayTracingTargetComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sMaterialComponent const &aComponent, string_t const &aMaterialPath );
    // void WriteComponent( ConfigurationWriter &aOut, sMaterialShaderComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sBackgroundComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sAmbientLightingComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sLightComponent const &aComponent );
    void WriteComponent( ConfigurationWriter &aOut, sUIComponent const &aComponent );
} // namespace SE::Core