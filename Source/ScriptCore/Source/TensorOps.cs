using System;

namespace SpockEngine
{
    public class sTensorShape
    {
        private ulong mInternalTensorShape;
        private bool mIsOwner;

        public sTensorShape()
        {
            mInternalTensorShape = 0;
            mIsOwner = true;
        }

        public sTensorShape(uint[,] aShape, uint aElementSize)
        {
            uint lLayers = (uint)aShape.GetLength(0);
            uint lRank = (uint)aShape.GetLength(1);

            mInternalTensorShape = CppCall.OpNode_NewTensorShape(aShape, lRank, lLayers, aElementSize);
            mIsOwner = true;
        }

        public sTensorShape(uint[] aShape, uint aElementSize)
        {
            var lShape = new uint[aShape.GetLength(0), 1];
            Buffer.BlockCopy(aShape, 0, lShape, 0, aShape.GetLength(0) * sizeof(uint));

            mInternalTensorShape = CppCall.OpNode_NewTensorShape(lShape, 1, (uint)aShape.GetLength(0), aElementSize);
            mIsOwner = true;
        }

        public sTensorShape(ulong aInternalTensorShape)
        {
            mInternalTensorShape = aInternalTensorShape;
            mIsOwner = false;
        }

        public uint CountLayers()
        {
            if (mInternalTensorShape == 0) return 0;

            return CppCall.OpNode_CountLayers(mInternalTensorShape);
        }

        public uint[] GetDimension(int i)
        {
            if (mInternalTensorShape == 0) return Array.Empty<uint>();

            return CppCall.OpNode_GetDimension(mInternalTensorShape, i);
        }

        public void Trim(int aToDimension)
        {
            if (mInternalTensorShape == 0) return;

            CppCall.OpNode_Trim(mInternalTensorShape, aToDimension);
        }

        public void Flatten(int aToDimension)
        {
            if (mInternalTensorShape == 0) return;

            CppCall.OpNode_Flatten(mInternalTensorShape, aToDimension);
        }

        public void InsertDimension(int aPosition, uint[] aDimension)
        {
            if (mInternalTensorShape == 0) return;

            CppCall.OpNode_InsertDimension(mInternalTensorShape, aPosition, aDimension);
        }

        ~sTensorShape()
        {
            if (mIsOwner)
                CppCall.OpNode_DestroyTensorShape(mInternalTensorShape);
        }
    };

    // public class Scope
    // {
    //     private ulong mInternalScope;
    //     private bool mIsOwner;

    //     public Scope(uint aMemorySize)
    //     {
    //         mInternalScope = CppCall.OpNode_NewScope(aMemorySize);
    //         mIsOwner = true;
    //     }

    //     public Scope(ulong aInternalScope)
    //     {
    //         mInternalScope = aInternalScope;
    //         mIsOwner = false;
    //     }

    //     public ~Scope()
    //     {
    //         if (mIsOwner)
    //             CppCall.OpNode_DestroyScope(mInternalScope);
    //     }

    //     public InternalScope() { return mInternalScope; }
    // };

    // public struct sConstantValueInitializerComponent<_Ty>
    // {
    //     private Type mType;
    //     private _Ty mValue;

    //     sConstantValueInitializerComponent(_Ty aValue) { mType = typeof(_Ty); mValue = aValue; }
    // };

    // public struct sVectorInitializerComponent<_Ty>
    // {
    //     private Type mType;
    //     private List<_Ty> mValue;

    //     sVectorInitializerComponent(_Ty[] aValue) { mType = typeof(_Ty); mValue = new List<_Ty>(aValue); }
    //     sVectorInitializerComponent(List<_Ty> aValue) { mType = typeof(_Ty); mValue = aValue; }
    // };

    // public struct sDataInitializerComponent<_Ty>
    // {
    //     private Type mType;
    //     private List<_Ty> mValue;

    //     sDataInitializerComponent(_Ty[] aValue) { mType = typeof(_Ty); mValue = new List<_Ty>(aValue); }
    //     sDataInitializerComponent(List<_Ty> aValue) { mType = typeof(_Ty); mValue = aValue; }
    // };

    // public struct sRandomUniformInitializerComponent<_Ty>
    // {
    //     private Type mType;
    // };

    // public struct sRandomNormalInitializerComponent<_Ty>
    // {
    //     private Type mType;
    //     private _Ty mMean;
    //     private _Ty mStd;
    // };

    // public struct OpNode
    // {

    //     private uint mEntityID;
    //     private Scope mScope;

    //     public OpNode(uint aEntityID, Scope aScope)
    //     {
    //         mEntityID = aEntityID;
    //         mScope = aScope;
    //     }

    //     public uint EntityID() { return mEntityID; }
    //     public Scope ScopeID() { return mScope; }
    // };

    // public class TensorOps
    // {
    //     OpNode MultiTensorValue(Scope aScope, sConstantValueInitializerComponent aInitializer, sTensorShape aShape)
    //     {

    //     }

    //     OpNode MultiTensorValue(Scope aScope, sVectorInitializerComponent aInitializer, sTensorShape aShape)
    //     {

    //     }

    //     OpNode MultiTensorValue(Scope aScope, sDataInitializerComponent aInitializer, sTensorShape aShape)
    //     {

    //     }

    //     OpNode MultiTensorValue(Scope aScope, sRandomUniformInitializerComponent aInitializer, sTensorShape aShape)
    //     {

    //     }

    //     OpNode MultiTensorValue(Scope aScope, sRandomNormalInitializerComponent aInitializer, sTensorShape aShape)
    //     {

    //     }

    //     OpNode VectorValue<_Ty>(Scope aScope, List<_Ty> aValue)
    //     {

    //     }

    //     OpNode ScalarVectorValue<_Ty>(Scope aScope, List<_Ty> aValue)
    //     {

    //     }

    //     OpNode ConstantScalarValue<_Ty>(Scope aScope, _Ty aValue)
    //     {

    //     }

    //     OpNode Add(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Add(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Subtract(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Subtract(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Divide(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Divide(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Multiply(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Multiply(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode And(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_And(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Or(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Or(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Not(Scope aScope, OpNode aOperand)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Not(aScope.InternalScope(), aOperand.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode BitwiseAnd(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_BitwiseAnd(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode BitwiseOr(Scope aScope, OpNode aLeft, OpNode aRight)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_BitwiseOr(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode BitwiseNot(Scope aScope, OpNode aOperand)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_BitwiseNot(aScope.InternalScope(), aOperand.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode InInterval(Scope aScope, OpNode aX, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_InInterval(aScope.InternalScope(), aLower.EntityID(), aUpper.EntityID(), aStrictLower, aStrictUpper);

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Equal(Scope aScope, OpNode aX, OpNode aY)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Equal(aScope.InternalScope(), aX.EntityID(), aY.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode LessThan(Scope aScope, OpNode aX, OpNode aY)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_LessThan(aScope.InternalScope(), aX.EntityID(), aY.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode LessThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_LessThanOrEqual(aScope.InternalScope(), aX.EntityID(), aY.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode GreaterThan(Scope aScope, OpNode aX, OpNode aY)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_GreaterThan(aScope.InternalScope(), aX.EntityID(), aY.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode GreaterThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_GreaterThanOrEqual(aScope.InternalScope(), aX.EntityID(), aY.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Where(Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Where(aScope.InternalScope(), aCondition.EntityID(), aValueIfTrue.EntityID(), aValueIfFalse.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Mix(Scope aScope, OpNode aA, OpNode aB, OpNode aT)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Mix(aScope.InternalScope(), aA.EntityID(), aB.EntityID(), aT.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode AffineTransform(Scope aScope, OpNode aA, OpNode aX, OpNode aB)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_AffineTransform(aScope.InternalScope(), aA.EntityID(), aX.EntityID(), aB.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode ARange(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_ARange(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID(), aDelta.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode LinearSpace(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_LinearSpace(aScope.InternalScope(), aLeft.EntityID(), aRight.EntityID(), aSubdivisions.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Repeat(Scope aScope, OpNode aArray, OpNode aRepetitions)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Repeat(aScope.InternalScope(), aArray.EntityID(), aRepetitions.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Tile(Scope aScope, OpNode aArray, OpNode aRepetitions)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Tile(aScope.InternalScope(), aArray.EntityID(), aRepetitions.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Sample2D(Scope aScope, OpNode aX, OpNode aY, OpNode aTextures)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Sample2D(aScope.InternalScope(), aX.EntityID(), aY.EntityID(), aTextures.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Collapse(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Collapse(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Expand(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Expand(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Reshape(Scope aScope, OpNode aArray, sTensorShape aNewShape)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Reshape(aScope.InternalScope(), aNewShape);

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Relayout(Scope aScope, OpNode aArray, sTensorShape aNewLayout)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Relayout(aScope.InternalScope(), aNewLayout);

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Flatten(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Flatten(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Slice(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Slice(aScope.InternalScope(), aArray.EntityID(), aBegin.EntityID(), aEnd.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Summation(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Summation(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Summation(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Summation(aScope.InternalScope(), aArray.EntityID(), aBegin.EntityID(), aEnd.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode CountTrue(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_CountTrue(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode CountNonZero(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_CountNonZero(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode CountZero(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_CountZero(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Floor(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Floor(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Ceil(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Ceil(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Abs(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Abs(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Sqrt(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Sqrt(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Round(Scope aScope, OpNode aArray)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Round(aScope.InternalScope(), aArray.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Diff(Scope aScope, OpNode aArray, UInt32 aCount)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Diff(aScope.InternalScope(), aArray.EntityID(), aCount);

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Shift(Scope aScope, OpNode aArray, Int32 aCount, OpNode aFillValue)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Shift(aScope.InternalScope(), aArray.EntityID(), aCount, aFillValue.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode Conv1D(Scope aScope, OpNode aArray0, OpNode aArray1)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_Conv1D(aScope.InternalScope(), aArray0.EntityID(), aArray1.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }

    //     OpNode HCat(Scope aScope, OpNode aArray0, OpNode aArray1)
    //     {
    //         ulong lNewOpNodeID = CppCall.OpNode_HCat(aScope.InternalScope(), aArray0.EntityID(), aArray1.EntityID());

    //         return OpNode(lNewOpNodeID, aScope);
    //     }
    // };
}