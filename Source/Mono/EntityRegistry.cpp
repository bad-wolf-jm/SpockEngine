#include "EntityRegistry.h"

// #include "TypeReflection.h"

// #include "mono/jit/jit.h"
// #include "mono/metadata/assembly.h"
// #include "mono/metadata/object.h"
// #include "mono/metadata/tabledefs.h"

#include <functional>
#include <string>

namespace LTSE::Core
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

} // namespace LTSE::Core