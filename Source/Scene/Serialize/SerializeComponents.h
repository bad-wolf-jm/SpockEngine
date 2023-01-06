
#include <string>
#include <unordered_map>

#include "Core/EntityRegistry/Registry.h"

#include "Scene/Components.h"
#include "Serialize/FileIO.h"

namespace SE::Core
{
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core::EntityComponentSystem::Components;

    using EntityMap = std::unordered_map<std::string, Entity>;

    void ReadComponent( sTag &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sCameraComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sActorComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sAnimationChooser &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sAnimationComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities,
                        std::vector<sImportedAnimationSampler> &aInterpolationData );
    void ReadComponent( sAnimatedTransformComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sNodeTransformComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sTransformMatrixComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sStaticMeshComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sParticleSystemComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sParticleShaderComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sWireframeComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sWireframeMeshComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sBoundingBoxComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sSkeletonComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sRayTracingTargetComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sMaterialComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sMaterialShaderComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sBackgroundComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sAmbientLightingComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );
    void ReadComponent( sLightComponent cons &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities );

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sTag const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sRelationshipComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sCameraComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAnimationChooser const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sActorComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAnimatedTransformComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sNodeTransformComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sTransformMatrixComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sStaticMeshComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sParticleSystemComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sParticleShaderComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sSkeletonComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sWireframeComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sWireframeMeshComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sBoundingBoxComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sRayTracingTargetComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sMaterialComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sMaterialShaderComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sBackgroundComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAmbientLightingComponent const &aComponent );
    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sLightComponent const &aComponent );
} // namespace SE::Core