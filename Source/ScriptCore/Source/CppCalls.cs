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
        internal extern static uint OpNode_CreateMultiTensor_Constant_Initializer(ref Scope aScope, Type aType, object aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Vector_Initializer<_Ty>(ref Scope aScope, sVectorInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Data_Initializer<_Ty>(ref Scope aScope, sDataInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Random_Normal_Initializer<_Ty>(ref Scope aScope, sRandomNormalInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateMultiTensor_Random_Uniform_Initializer<_Ty>(ref Scope aScope, sRandomUniformInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateVector<_Ty>(ref Scope aScope, _Ty[] aInitializer);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateScalarVector<_Ty>(ref Scope aScope, _Ty[] aInitializer);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CreateScalarValue<_Ty>(ref Scope aScope, _Ty aInitializer);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Add(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Subtract(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Divide(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Multiply(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_And(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Or(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Not(ref Scope aScope, OpNode aOperand);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseAnd(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseOr(ref Scope aScope, OpNode aLeft, OpNode aRight);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_BitwiseNot(ref Scope aScope, OpNode aOperand);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_InInterval(ref Scope aScope, OpNode aX, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Equal(ref Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LessThan(ref Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LessThanOrEqual(ref Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_GreaterThan(ref Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_GreaterThanOrEqual(ref Scope aScope, OpNode aX, OpNode aY);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Where(ref Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Mix(ref Scope aScope, OpNode aA, OpNode aB, OpNode aT);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_AffineTransform(ref Scope aScope, OpNode aA, OpNode aX, OpNode aB);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_ARange(ref Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_LinearSpace(ref Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Repeat(ref Scope aScope, OpNode aArray, OpNode aRepetitions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Tile(ref Scope aScope, OpNode aArray, OpNode aRepetitions);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Sample2D(ref Scope aScope, OpNode aX, OpNode aY, OpNode aTextures);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Collapse(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Expand(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Reshape(ref Scope aScope, OpNode aArray, sTensorShape aNewShape);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Relayout(ref Scope aScope, OpNode aArray, sTensorShape aNewLayout);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_FlattenNode(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Slice(ref Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Summation(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Summation(ref Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountTrue(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountNonZero(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountZero(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Floor(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Ceil(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Abs(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Sqrt(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Round(ref Scope aScope, OpNode aArray);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Diff(ref Scope aScope, OpNode aArray, uint aCount);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Shift(ref Scope aScope, OpNode aArray, int aCount, OpNode aFillValue);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_Conv1D(ref Scope aScope, OpNode aArray0, OpNode aArray1);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_HCat(ref Scope aScope, OpNode aArray0, OpNode aArray1);
        
    }
}