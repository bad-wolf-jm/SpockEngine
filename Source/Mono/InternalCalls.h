#pragma once

#include "Core/EntityRegistry/Registry.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#define SE_ADD_INTERNAL_CALL( Name ) mono_add_internal_call( "SpockEngine.CppCall::" #Name, Name )

namespace SE::MonoInternalCalls
{
    using namespace SE::Core;

    uint32_t Entity_Create( EntityRegistry *aRegistry, MonoString *aName, uint32_t aParentEntityID );

    bool Entity_IsValid( uint32_t aEntityID, EntityRegistry *aRegistry );

    bool Entity_Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType );

    MonoObject *Entity_Get( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType );

    void Entity_Add( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent );

    void Entity_Remove( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType );

    void Entity_Replace( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType,
                         MonoObject *aNewComponent );

} // namespace SE::MonoInternalCalls