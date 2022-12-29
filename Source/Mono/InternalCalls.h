#pragma once

#include "Core/EntityRegistry/Registry.h"

#include "Core/CUDA/Array/MultiTensor.h"

#include "TensorOps/Scope.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#define SE_ADD_INTERNAL_CALL( Name ) mono_add_internal_call( "SpockEngine.CppCall::" #Name, Name )

namespace SE::MonoInternalCalls
{
    using namespace SE::Core;
    using namespace SE::TensorOps;

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

    size_t OpNode_NewScope( uint32_t aMemorySize );
    void   OpNode_DestroyScope( Scope *aTensorShape );

    uint32_t OpNode_CreateMultiTensor_Constant_Initializer( MonoObject *aScope, MonoObject *aInitializer, MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Vector_Initializer( MonoObject *aScope, MonoObject *aInitializer, MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Data_Initializer( MonoObject *aScope, MonoObject *aInitializer, MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Random_Uniform_Initializer( MonoObject *aScope, MonoObject *aInitializer, MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Random_Normal_Initializer( MonoObject *aScope, MonoObject *aInitializer, MonoObject *aShape );
    uint32_t OpNode_CreateVector( MonoObject *aScope, MonoArray *aValues );
    uint32_t OpNode_CreateScalarVector( MonoObject *aScope, MonoArray *aValues );
    uint32_t OpNode_CreateScalarValue( MonoObject *aScope, MonoObject *aInitializer );

    uint32_t OpNode_Add( MonoObject *aScope );
    uint32_t OpNode_Subtract( MonoObject *aScope );
    uint32_t OpNode_Divide( MonoObject *aScope );
    uint32_t OpNode_Multiply( MonoObject *aScope );
    uint32_t OpNode_And( MonoObject *aScope );
    uint32_t OpNode_Or( MonoObject *aScope );
    uint32_t OpNode_Not( MonoObject *aScope );
    uint32_t OpNode_BitwiseAnd( MonoObject *aScope );
    uint32_t OpNode_BitwiseOr( MonoObject *aScope );
    uint32_t OpNode_BitwiseNot( MonoObject *aScope );
    uint32_t OpNode_InInterval( MonoObject *aScope );
    uint32_t OpNode_LessThan( MonoObject *aScope );
    uint32_t OpNode_LessThanOrEqual( MonoObject *aScope );
    uint32_t OpNode_GreaterThan( MonoObject *aScope );
    uint32_t OpNode_GreaterThanOrEqual( MonoObject *aScope );
    uint32_t OpNode_Where( MonoObject *aScope );
    uint32_t OpNode_Mix( MonoObject *aScope );
    uint32_t OpNode_AffineTransform( MonoObject *aScope );
    uint32_t OpNode_LinearSpace( MonoObject *aScope );
    uint32_t OpNode_Repeat( MonoObject *aScope );
    uint32_t OpNode_Tile( MonoObject *aScope );
    uint32_t OpNode_Sample2D( MonoObject *aScope );
    uint32_t OpNode_Collapse( MonoObject *aScope );
    uint32_t OpNode_Expand( MonoObject *aScope );
    uint32_t OpNode_Reshape( MonoObject *aScope );
    uint32_t OpNode_Relayout( MonoObject *aScope );
    uint32_t OpNode_Flatten( MonoObject *aScope );
    uint32_t OpNode_Slice( MonoObject *aScope );
    uint32_t OpNode_Summation( MonoObject *aScope );
    uint32_t OpNode_CountTrue( MonoObject *aScope );
    uint32_t OpNode_CountNonZero( MonoObject *aScope );
    uint32_t OpNode_CountZero( MonoObject *aScope );
    uint32_t OpNode_Floor( MonoObject *aScope );
    uint32_t OpNode_Ceil( MonoObject *aScope );
    uint32_t OpNode_Abs( MonoObject *aScope );
    uint32_t OpNode_Sqrt( MonoObject *aScope );
    uint32_t OpNode_Round( MonoObject *aScope );
    uint32_t OpNode_Diff( MonoObject *aScope );
    uint32_t OpNode_Shift( MonoObject *aScope );
    uint32_t OpNode_Conv1D( MonoObject *aScope );
    uint32_t OpNode_HCat( MonoObject *aScope );

} // namespace SE::MonoInternalCalls