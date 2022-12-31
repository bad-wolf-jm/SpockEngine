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

        public void InsertDimension(int aPosition, ref uint[] aDimension)
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

    public class Scope
    {
        private ulong mInternalScope;
        private bool mIsOwner;

        public Scope()
        {
            mInternalScope = 0;
            mIsOwner = true;
        }

        public Scope(uint aMemorySize)
        {
            mInternalScope = CppCall.OpNode_NewScope(aMemorySize);
            mIsOwner = true;
        }

        public Scope(ulong aInternalScope, bool aDummy)
        {
            mInternalScope = aInternalScope;
            mIsOwner = false;
        }

        ~Scope()
        {
            if (mIsOwner)
                CppCall.OpNode_DestroyScope(mInternalScope);
        }
    };

    public class sConstantValueInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty mValue;

        public sConstantValueInitializerComponent(_Ty aValue) { mType = typeof(_Ty); mValue = aValue; }
        public sConstantValueInitializerComponent(sConstantValueInitializerComponent<_Ty> aValue) { mType = aValue.mType; mValue = aValue.mValue; }

        public Type Type() { return mType; }
        public _Ty Value() { return mValue; }
    };

    public class sVectorInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty[] mValue;

        public sVectorInitializerComponent(ref _Ty[] aValue) { mType = typeof(_Ty); mValue = aValue; }
    };

    public class sDataInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty[] mValue;

        public sDataInitializerComponent(ref _Ty[] aValue) { mType = typeof(_Ty); mValue = aValue; }
    };

    public class sRandomUniformInitializerComponent<_Ty>
    {
        private Type mType;

        public sRandomUniformInitializerComponent() { mType = typeof(_Ty); }
    };

    public class sRandomNormalInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty mMean;
        private _Ty mStd;

        public sRandomNormalInitializerComponent(_Ty aMean, _Ty aStd) { mType = typeof(_Ty); mMean = aMean; mStd = aStd; }
    };

    public class OpNode
    {

        private uint mEntityID;
        private Scope mScope;

        public OpNode() { mEntityID = 0; mScope = new Scope(); }

        public OpNode(uint aEntityID, ref Scope aScope) { mEntityID = aEntityID; mScope = aScope; }
    };

    public class TensorOps
    {
        public static OpNode MultiTensorValue<_Ty>(ref Scope aScope, sConstantValueInitializerComponent<_Ty> aInitializer, ref sTensorShape aShape)
        {
            object lBoxedValue = aInitializer.Value();
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Constant_Initializer(ref aScope, aInitializer.Type(), lBoxedValue, ref aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(ref Scope aScope, sVectorInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Vector_Initializer(ref aScope, aInitializer, ref aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(ref Scope aScope, sDataInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Data_Initializer(ref aScope, aInitializer, ref aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(ref Scope aScope, sRandomUniformInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Random_Uniform_Initializer(ref aScope, aInitializer, ref aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(ref Scope aScope, sRandomNormalInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Random_Normal_Initializer(ref aScope, aInitializer, ref aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode VectorValue<_Ty>(ref Scope aScope, _Ty[] aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateVector(ref aScope, aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode ScalarVectorValue<_Ty>(ref Scope aScope, _Ty[] aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateScalarVector(ref aScope, aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode ConstantScalarValue<_Ty>(ref Scope aScope, _Ty aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateScalarValue(ref aScope, aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode Add(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Add(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Subtract(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Subtract(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Divide(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Divide(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Multiply(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Multiply(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode And(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_And(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Or(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Or(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Not(ref Scope aScope, OpNode aOperand)
        {
            uint lNewOpNodeID = CppCall.OpNode_Not(ref aScope, aOperand);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseAnd(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseAnd(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseOr(ref Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseOr(ref aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseNot(ref Scope aScope, OpNode aOperand)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseNot(ref aScope, aOperand);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode InInterval(ref Scope aScope, OpNode aX, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper)
        {
            uint lNewOpNodeID = CppCall.OpNode_InInterval(ref aScope, aX, aLower, aUpper, aStrictLower, aStrictUpper);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Equal(ref Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_Equal(ref aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LessThan(ref Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_LessThan(ref aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LessThanOrEqual(ref Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_LessThanOrEqual(ref aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode GreaterThan(ref Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_GreaterThan(ref aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode GreaterThanOrEqual(ref Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_GreaterThanOrEqual(ref aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Where(ref Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse)
        {
            uint lNewOpNodeID = CppCall.OpNode_Where(ref aScope, aCondition, aValueIfTrue, aValueIfFalse);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Mix(ref Scope aScope, OpNode aA, OpNode aB, OpNode aT)
        {
            uint lNewOpNodeID = CppCall.OpNode_Mix(ref aScope, aA, aB, aT);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode AffineTransform(ref Scope aScope, OpNode aA, OpNode aX, OpNode aB)
        {
            uint lNewOpNodeID = CppCall.OpNode_AffineTransform(ref aScope, aA, aX, aB);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode ARange(ref Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta)
        {
            uint lNewOpNodeID = CppCall.OpNode_ARange(ref aScope, aLeft, aRight, aDelta);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LinearSpace(ref Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions)
        {
            uint lNewOpNodeID = CppCall.OpNode_LinearSpace(ref aScope, aLeft, aRight, aSubdivisions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Repeat(ref Scope aScope, OpNode aArray, OpNode aRepetitions)
        {
            uint lNewOpNodeID = CppCall.OpNode_Repeat(ref aScope, aArray, aRepetitions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Tile(ref Scope aScope, OpNode aArray, OpNode aRepetitions)
        {
            uint lNewOpNodeID = CppCall.OpNode_Tile(ref aScope, aArray, aRepetitions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Sample2D(ref Scope aScope, OpNode aX, OpNode aY, OpNode aTextures)
        {
            uint lNewOpNodeID = CppCall.OpNode_Sample2D(ref aScope, aX, aY, aTextures);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Collapse(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Collapse(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Expand(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Expand(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Reshape(ref Scope aScope, OpNode aArray, sTensorShape aNewShape)
        {
            uint lNewOpNodeID = CppCall.OpNode_Reshape(ref aScope, aArray, aNewShape);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Relayout(ref Scope aScope, OpNode aArray, sTensorShape aNewLayout)
        {
            uint lNewOpNodeID = CppCall.OpNode_Relayout(ref aScope, aArray, aNewLayout);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Flatten(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_FlattenNode(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Slice(ref Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {
            uint lNewOpNodeID = CppCall.OpNode_Slice(ref aScope, aArray, aBegin, aEnd);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Summation(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Summation(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Summation(ref Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {
            uint lNewOpNodeID = CppCall.OpNode_Summation(ref aScope, aArray, aBegin, aEnd);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountTrue(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountTrue(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountNonZero(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountNonZero(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountZero(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountZero(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Floor(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Floor(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Ceil(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Ceil(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Abs(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Abs(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Sqrt(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Sqrt(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Round(ref Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Round(ref aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Diff(ref Scope aScope, OpNode aArray, UInt32 aCount)
        {
            uint lNewOpNodeID = CppCall.OpNode_Diff(ref aScope, aArray, aCount);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Shift(ref Scope aScope, OpNode aArray, int aCount, OpNode aFillValue)
        {
            uint lNewOpNodeID = CppCall.OpNode_Shift(ref aScope, aArray, aCount, aFillValue);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Conv1D(ref Scope aScope, OpNode aArray0, OpNode aArray1)
        {
            uint lNewOpNodeID = CppCall.OpNode_Conv1D(ref aScope, aArray0, aArray1);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode HCat(ref Scope aScope, OpNode aArray0, OpNode aArray1)
        {
            uint lNewOpNodeID = CppCall.OpNode_HCat(ref aScope, aArray0, aArray1);

            return new OpNode(lNewOpNodeID, ref aScope);
        }
    };
}