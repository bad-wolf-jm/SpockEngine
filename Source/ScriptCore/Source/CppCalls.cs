using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public static class CppCall
    {
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint Entity_Create(ulong aRegistry, string aName, uint aParentEntityID);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_IsValid(uint aEntityID, ulong aRegistry);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_Has(uint aEntityID, ulong aRegistry, Type aTypeDesc);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static T Entity_Get<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Add<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Replace<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Remove(uint aEntityID, ulong aRegistry, Type aTypeDesc);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_NewTensorShape(uint[,] aShape, uint aRank, uint aLayers, uint aElementSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_DestroyTensorShape(ulong aTensorShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountLayers(ulong aTensorShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint[] OpNode_GetDimension(ulong aTensorShape, int i);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_Trim(ulong aTensorShape, int aToDimension);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_Flatten(ulong aTensorShape, int aToDimension);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_InsertDimension(ulong aTensorShape, int aPosition, uint[] aDimension);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_NewScope(uint aMemorySize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_DestroyScope(ulong aScope);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Constant_Initializer<_Ty>(Scope aScope, sConstantValueInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Vector_Initializer<_Ty>(Scope aScope, sVectorInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Data_Initializer<_Ty>(Scope aScope, sDataInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Random_Normal_Initializer<_Ty>(Scope aScope, sRandomNormalInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Random_Uniform_Initializer<_Ty>(Scope aScope, sRandomUniformInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateVector<_Ty>(Scope aScope, _Ty[] aInitializer);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateScalarVector<_Ty>(Scope aScope, _Ty[] aInitializer);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateScalarValue<_Ty>(Scope aScope, _Ty aInitializer);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Add(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Subtract(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Divide(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Multiply(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_And(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Or(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Not(Scope aScope, OpNode aOperand);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseAnd(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseOr(Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseNot(Scope aScope, OpNode aOperand);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_InInterval(Scope aScope, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Equal(Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LessThan(Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LessThanOrEqual(Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_GreaterThan(Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_GreaterThanOrEqual(Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Where(Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Mix(Scope aScope, OpNode aA, OpNode aB, OpNode aT);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_AffineTransform(Scope aScope, OpNode aA, OpNode aX, OpNode aB);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_ARange(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LinearSpace(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Repeat(Scope aScope, OpNode aArray, OpNode aRepetitions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Tile(Scope aScope, OpNode aArray, OpNode aRepetitions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Sample2D(Scope aScope, OpNode aX, OpNode aY, OpNode aTextures);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Collapse(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Expand(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Reshape(Scope aScope, sTensorShape aNewShape);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Relayout(Scope aScope, sTensorShape aNewLayout);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_FlattenNode(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Slice(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Summation(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Summation(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountTrue(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountNonZero(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountZero(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Floor(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Ceil(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Abs(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Sqrt(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Round(Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Diff(Scope aScope, OpNode aArray, uint aCount);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Shift(Scope aScope, OpNode aArray, int aCount, OpNode aFillValue);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Conv1D(Scope aScope, OpNode aArray0, OpNode aArray1);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_HCat(Scope aScope, OpNode aArray0, OpNode aArray1);
        
    }
}