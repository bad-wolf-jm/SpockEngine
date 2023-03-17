#pragma once

#include "Core/EntityCollection/Collection.h"

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

    // void Console_Initialize();
    // void Console_Write( MonoString *aBuffer );

    uint32_t Entity_Create( EntityCollection *aRegistry, MonoString *aName, uint32_t aParentEntityID );

    bool Entity_IsValid( uint32_t aEntityID, EntityCollection *aRegistry );

    bool Entity_Has( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType );

    MonoObject *Entity_Get( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType );

    void Entity_Add( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent );

    void Entity_Remove( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType );

    void Entity_Replace( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType,
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

    uint32_t OpNode_CreateMultiTensor_Constant_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aInitializer,
                                                            MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Vector_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aInitializer,
                                                          MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Data_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aInitializer,
                                                        MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Random_Uniform_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aShape );
    uint32_t OpNode_CreateMultiTensor_Random_Normal_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aMean,
                                                                 MonoObject *aStd, MonoObject *aShape );
    uint32_t OpNode_CreateVector( MonoObject *aScope, MonoArray *aValues );
    uint32_t OpNode_CreateScalarVector( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aValues );
    uint32_t OpNode_CreateScalarValue( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aInitializer );

    uint32_t OpNode_Add( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_Subtract( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_Divide( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_Multiply( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_And( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_Or( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_Not( MonoObject *aScope, MonoObject *aOperand );
    uint32_t OpNode_BitwiseAnd( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_BitwiseOr( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight );
    uint32_t OpNode_BitwiseNot( MonoObject *aScope, MonoObject *aOperand );
    uint32_t OpNode_InInterval( MonoObject *aScope, MonoObject *aX, MonoObject *aLower, MonoObject *aUpper, bool aStrictLower,
                                bool aStrictUpper );
    uint32_t OpNode_Equal( MonoObject *aScope, MonoObject *aX, MonoObject *aY );
    uint32_t OpNode_LessThan( MonoObject *aScope, MonoObject *aX, MonoObject *aY );
    uint32_t OpNode_LessThanOrEqual( MonoObject *aScope, MonoObject *aX, MonoObject *aY );
    uint32_t OpNode_GreaterThan( MonoObject *aScope, MonoObject *aX, MonoObject *aY );
    uint32_t OpNode_GreaterThanOrEqual( MonoObject *aScope, MonoObject *aX, MonoObject *aY );
    uint32_t OpNode_Where( MonoObject *aScope, MonoObject *aCondition, MonoObject *aValueIfTrue, MonoObject *aValueIfFalse );
    uint32_t OpNode_Mix( MonoObject *aScope, MonoObject *aA, MonoObject *aB, MonoObject *aT );
    uint32_t OpNode_AffineTransform( MonoObject *aScope, MonoObject *aA, MonoObject *aX, MonoObject *aB );
    uint32_t OpNode_ARange( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight, MonoObject *aDelta );
    uint32_t OpNode_LinearSpace( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight, MonoObject *aSubdivisions );
    uint32_t OpNode_Repeat( MonoObject *aScope, MonoObject *aArray, MonoObject *aRepetitions );
    uint32_t OpNode_Tile( MonoObject *aScope, MonoObject *aArray, MonoObject *aRepetitions );
    uint32_t OpNode_Sample2D( MonoObject *aScope, MonoObject *aX, MonoObject *aY, MonoObject *aTextures );
    uint32_t OpNode_Collapse( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Expand( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Reshape( MonoObject *aScope, MonoObject *aArray, MonoObject *aNewShape );
    uint32_t OpNode_Relayout( MonoObject *aScope, MonoObject *aArray, MonoObject *aNewShape );
    uint32_t OpNode_FlattenNode( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Slice( MonoObject *aScope, MonoObject *aArray, MonoObject *aBegin, MonoObject *aEnd );
    uint32_t OpNode_Summation( MonoObject *aScope, MonoObject *aArray, MonoObject *aBegin, MonoObject *aEnd );
    uint32_t OpNode_CountTrue( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_CountNonZero( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_CountZero( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Floor( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Ceil( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Abs( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Sqrt( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Round( MonoObject *aScope, MonoObject *aArray );
    uint32_t OpNode_Diff( MonoObject *aScope, MonoObject *aArray, uint32_t aCount );
    uint32_t OpNode_Shift( MonoObject *aScope, MonoObject *aArray, int32_t aCount, MonoObject *aFillValue );
    uint32_t OpNode_Conv1D( MonoObject *aScope, MonoObject *aArray0, MonoObject *aArray1 );
    uint32_t OpNode_HCat( MonoObject *aScope, MonoObject *aArray0, MonoObject *aArray1 );

    void UI_Text( MonoString *aText );
    bool UI_Button( MonoString *aText );

} // namespace SE::MonoInternalCalls