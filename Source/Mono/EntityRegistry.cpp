#include "EntityRegistry.h"

// #include "TypeReflection.h"

// #include "mono/jit/jit.h"
// #include "mono/metadata/assembly.h"
// #include "mono/metadata/object.h"
// #include "mono/metadata/tabledefs.h"

#include <functional>
#include <string>

namespace SE::Core
{

    entt::meta_type GetMetaType( MonoType *aObject )
    {
        auto lHashValue = std::hash<uint64_t>()( (uint64_t)aObject );

        return entt::resolve( (uint32_t)( lHashValue & 0xFFFFFFFF ) );
    }

    ScriptClassInstance MarshallComponent( ScriptClass &lMonoType, sNodeTransformComponent &aComponent )
    {
        return lMonoType.Instantiate( aComponent.mMatrix );
    }

    void UnmarshallComponent( ScriptClassInstance &aMonoType, sNodeTransformComponent &aComponent )
    {
        math::mat4 lFieldValue = aMonoType.GetFieldValue<math::mat4>( "mMatrix" );

        aComponent = sNodeTransformComponent( lFieldValue );
    }

    ScriptClassInstance MarshallComponent( ScriptClass &lMonoType, sTransformMatrixComponent &aComponent )
    {
        return lMonoType.Instantiate( aComponent.Matrix );
    }

    void UnmarshallComponent( ScriptClassInstance &aMonoType, sTransformMatrixComponent &aComponent )
    {
        math::mat4 lFieldValue = aMonoType.GetFieldValue<math::mat4>( "mMatrix" );

        aComponent = sTransformMatrixComponent( lFieldValue );
    }

    ScriptClassInstance MarshallComponent( ScriptClass &lMonoType, sTag &aComponent )
    {
        MonoString *lManagedSTagValue = ScriptManager::NewString( aComponent.mValue );

        auto lNewObject = lMonoType.Instantiate( lManagedSTagValue );

        return lNewObject;
    }

    void UnmarshallComponent( ScriptClassInstance &aMonoType, sTag &aComponent )
    {
        auto lFieldValue = aMonoType.GetFieldValue<MonoString *>( "mValue" );

        aComponent = sTag( ScriptManager::NewString( lFieldValue ) );
    }

    ScriptClassInstance MarshallComponent( ScriptClass &lMonoType, sLightComponent &aComponent )
    {
        auto lNewObject = lMonoType.Instantiate( aComponent.mType, aComponent.mIntensity, aComponent.mColor, aComponent.mCone );

        return lNewObject;
    }

    void UnmarshallComponent( ScriptClassInstance &aMonoType, sLightComponent &aComponent )
    {
        aComponent.mType      = aMonoType.GetFieldValue<eLightType>( "mType" );
        aComponent.mIntensity = aMonoType.GetFieldValue<float>( "mIntensity" );
        aComponent.mColor     = aMonoType.GetFieldValue<math::vec3>( "mColor" );
        aComponent.mCone      = aMonoType.GetFieldValue<float>( "mCone" );
    }

} // namespace SE::Core