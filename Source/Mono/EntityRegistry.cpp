#include "EntityRegistry.h"

#include <functional>
#include <string>

namespace SE::Core
{

    entt::meta_type GetMetaType( MonoType *aObject )
    {
        auto lHashValue = std::hash<uint64_t>()( (uint64_t)aObject );

        return entt::resolve( (uint32_t)( lHashValue & 0xFFFFFFFF ) );
    }

    MonoScriptInstance MarshallComponent( MonoScriptClass &lMonoType, sNodeTransformComponent &aComponent )
    {
        return lMonoType.Instantiate( aComponent.mMatrix );
    }

    void UnmarshallComponent( MonoScriptInstance &aMonoType, sNodeTransformComponent &aComponent )
    {
        math::mat4 lFieldValue = aMonoType.GetFieldValue<math::mat4>( "mMatrix" );

        aComponent = sNodeTransformComponent( lFieldValue );
    }

    MonoScriptInstance MarshallComponent( MonoScriptClass &lMonoType, sTag &aComponent )
    {
        MonoString *lManagedSTagValue = MonoScriptEngine::NewString( aComponent.mValue );

        auto lNewObject = lMonoType.Instantiate( lManagedSTagValue );

        return lNewObject;
    }

    void UnmarshallComponent( MonoScriptInstance &aMonoType, sTag &aComponent )
    {
        auto lFieldValue = aMonoType.GetFieldValue<MonoString *>( "mValue" );

        aComponent = sTag( MonoScriptEngine::NewString( lFieldValue ) );
    }

    MonoScriptInstance MarshallComponent( MonoScriptClass &lMonoType, sLightComponent &aComponent )
    {
        auto lNewObject = lMonoType.Instantiate( aComponent.mType, aComponent.mIntensity, aComponent.mColor, aComponent.mCone );

        return lNewObject;
    }

    void UnmarshallComponent( MonoScriptInstance &aMonoType, sLightComponent &aComponent )
    {
        aComponent.mType      = aMonoType.GetFieldValue<eLightType>( "mType" );
        aComponent.mIntensity = aMonoType.GetFieldValue<float>( "mIntensity" );
        aComponent.mColor     = aMonoType.GetFieldValue<math::vec3>( "mColor" );
        aComponent.mCone      = aMonoType.GetFieldValue<float>( "mCone" );
    }

} // namespace SE::Core