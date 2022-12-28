#pragma once

#include "Core/EntityRegistry/Registry.h"

#include "Core/CUDA/Array/MultiTensor.h"

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

    size_t     OpNode_NewTensorShape( MonoArray *aShape, uint32_t aRank, uint32_t aLayers, uint32_t aElementSize );
    void       OpNode_DestroyTensorShape( Cuda::sTensorShape *aTensorShape );
    uint32_t   OpNode_CountLayers( Cuda::sTensorShape *aTensorShape );
    MonoArray *OpNode_GetDimension( Cuda::sTensorShape *aTensorShape, int i );
    void       OpNode_Trim( Cuda::sTensorShape *aTensorShape, int aToDimension );
    void       OpNode_Flatten( Cuda::sTensorShape *aTensorShape, int aToDimension );
    void       OpNode_InsertDimension( Cuda::sTensorShape *aTensorShape, int aPosition, MonoArray *aDimension );

} // namespace SE::MonoInternalCalls