#pragma once

#include "Core/EntityRegistry/Registry.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#define SE_ADD_INTERNAL_CALL( Name ) mono_add_internal_call( "SpockEngine.CppCall::" #Name, Name )

namespace LTSE::MonoInternalCalls
{
    using namespace LTSE::Core;

    void NativeLog( MonoString *string, int parameter );

    bool Entity_IsValid( uint32_t aEntityID, EntityRegistry *aRegistry );
    bool Entity_Has( uint32_t aEntityID, EntityRegistry *aRegistry, MonoReflectionType *aComponentType );

} // namespace LTSE::MonoInternalCalls