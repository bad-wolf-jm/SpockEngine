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
            Console.WriteLine("~sTensorShape()");
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
            Console.WriteLine("~Scope()");
            if (mIsOwner)
                CppCall.OpNode_DestroyScope(mInternalScope);
        }
    };

    public class sConstantValueInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty mValue;

        public sConstantValueInitializerComponent(_Ty aValue) { mType = typeof(_Ty); mValue = aValue; }

        public Type Type() { return mType; }

        public object Value() { return (object)mValue; }
    };

    public class sVectorInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty[] mValue;

        public sVectorInitializerComponent(ref _Ty[] aValue) { mType = typeof(_Ty); mValue = aValue; }

        public Type Type() { return mType; }
        public _Ty[] Value() { return mValue; }
    };

    public class sDataInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty[] mValue;

        public sDataInitializerComponent(ref _Ty[] aValue) { mType = typeof(_Ty); mValue = aValue; }

        public Type Type() { return mType; }
        public _Ty[] Value() { return mValue; }
    };

    public class sRandomUniformInitializerComponent<_Ty>
    {
        private Type mType;

        public sRandomUniformInitializerComponent() { mType = typeof(_Ty); }

        public Type Type() { return mType; }
    };

    public class sRandomNormalInitializerComponent<_Ty>
    {
        private Type mType;
        private _Ty mMean;
        private _Ty mStd;

        public sRandomNormalInitializerComponent(_Ty aMean, _Ty aStd) { mType = typeof(_Ty); mMean = aMean; mStd = aStd; }

        public Type Type() { return mType; }
        public object Mean() { return (object)mMean; }
        public object Std() { return (object)mStd; }

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
        public static OpNode MultiTensorValue<_Ty>(Scope aScope, sConstantValueInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Constant_Initializer(aScope, aInitializer.Type(), aInitializer.Value(), aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(Scope aScope, sVectorInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Vector_Initializer(aScope, aInitializer.Type(), aInitializer.Value(), aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(Scope aScope, sDataInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Data_Initializer(aScope, aInitializer.Type(), aInitializer.Value(), aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(Scope aScope, sRandomUniformInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Random_Uniform_Initializer(aScope, aInitializer.Type(), aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode MultiTensorValue<_Ty>(Scope aScope, sRandomNormalInitializerComponent<_Ty> aInitializer, sTensorShape aShape)
        {
            var lNodeHandle = CppCall.OpNode_CreateMultiTensor_Random_Normal_Initializer(aScope, aInitializer.Type(), aInitializer.Mean(), aInitializer.Std(), aShape);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode VectorValue<_Ty>(Scope aScope, _Ty[] aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateVector(aScope, aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode ScalarVectorValue<_Ty>(Scope aScope, _Ty[] aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateScalarVector(aScope, typeof(_Ty), aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode ConstantScalarValue<_Ty>(Scope aScope, _Ty aValue)
        {
            var lNodeHandle = CppCall.OpNode_CreateScalarValue(aScope, typeof(_Ty), (object)aValue);

            return new OpNode(lNodeHandle, ref aScope);
        }

        public static OpNode Add(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Add(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Subtract(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Subtract(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Divide(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Divide(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Multiply(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Multiply(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode And(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_And(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Or(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_Or(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Not(Scope aScope, OpNode aOperand)
        {
            uint lNewOpNodeID = CppCall.OpNode_Not(aScope, aOperand);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseAnd(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseAnd(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseOr(Scope aScope, OpNode aLeft, OpNode aRight)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseOr(aScope, aLeft, aRight);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode BitwiseNot(Scope aScope, OpNode aOperand)
        {
            uint lNewOpNodeID = CppCall.OpNode_BitwiseNot(aScope, aOperand);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode InInterval(Scope aScope, OpNode aX, OpNode aLower, OpNode aUpper, bool aStrictLower, bool aStrictUpper)
        {
            uint lNewOpNodeID = CppCall.OpNode_InInterval(aScope, aX, aLower, aUpper, aStrictLower, aStrictUpper);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Equal(Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_Equal(aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LessThan(Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_LessThan(aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LessThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_LessThanOrEqual(aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode GreaterThan(Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_GreaterThan(aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode GreaterThanOrEqual(Scope aScope, OpNode aX, OpNode aY)
        {
            uint lNewOpNodeID = CppCall.OpNode_GreaterThanOrEqual(aScope, aX, aY);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Where(Scope aScope, OpNode aCondition, OpNode aValueIfTrue, OpNode aValueIfFalse)
        {
            uint lNewOpNodeID = CppCall.OpNode_Where(aScope, aCondition, aValueIfTrue, aValueIfFalse);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Mix(Scope aScope, OpNode aA, OpNode aB, OpNode aT)
        {
            uint lNewOpNodeID = CppCall.OpNode_Mix(aScope, aA, aB, aT);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode AffineTransform(Scope aScope, OpNode aA, OpNode aX, OpNode aB)
        {
            uint lNewOpNodeID = CppCall.OpNode_AffineTransform(aScope, aA, aX, aB);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode ARange(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aDelta)
        {
            uint lNewOpNodeID = CppCall.OpNode_ARange(aScope, aLeft, aRight, aDelta);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode LinearSpace(Scope aScope, OpNode aLeft, OpNode aRight, OpNode aSubdivisions)
        {
            uint lNewOpNodeID = CppCall.OpNode_LinearSpace(aScope, aLeft, aRight, aSubdivisions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Repeat(Scope aScope, OpNode aArray, OpNode aRepetitions)
        {
            uint lNewOpNodeID = CppCall.OpNode_Repeat(aScope, aArray, aRepetitions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Tile(Scope aScope, OpNode aArray, OpNode aRepetitions)
        {
            uint lNewOpNodeID = CppCall.OpNode_Tile(aScope, aArray, aRepetitions);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Sample2D(Scope aScope, OpNode aX, OpNode aY, OpNode aTextures)
        {
            uint lNewOpNodeID = CppCall.OpNode_Sample2D(aScope, aX, aY, aTextures);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Collapse(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Collapse(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Expand(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Expand(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Reshape(Scope aScope, OpNode aArray, sTensorShape aNewShape)
        {
            uint lNewOpNodeID = CppCall.OpNode_Reshape(aScope, aArray, aNewShape);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Relayout(Scope aScope, OpNode aArray, sTensorShape aNewLayout)
        {
            uint lNewOpNodeID = CppCall.OpNode_Relayout(aScope, aArray, aNewLayout);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Flatten(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_FlattenNode(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Slice(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {
            uint lNewOpNodeID = CppCall.OpNode_Slice(aScope, aArray, aBegin, aEnd);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Summation(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Summation(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Summation(Scope aScope, OpNode aArray, OpNode aBegin, OpNode aEnd)
        {
            uint lNewOpNodeID = CppCall.OpNode_Summation(aScope, aArray, aBegin, aEnd);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountTrue(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountTrue(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountNonZero(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountNonZero(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode CountZero(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_CountZero(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Floor(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Floor(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Ceil(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Ceil(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Abs(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Abs(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Sqrt(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Sqrt(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Round(Scope aScope, OpNode aArray)
        {
            uint lNewOpNodeID = CppCall.OpNode_Round(aScope, aArray);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        public static OpNode Diff(Scope aScope, OpNode aArray, UInt32 aCount)
        {
            uint lNewOpNodeID = CppCall.OpNode_Diff(aScope, aArray, aCount);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Shift(Scope aScope, OpNode aArray, int aCount, OpNode aFillValue)
        {
            uint lNewOpNodeID = CppCall.OpNode_Shift(aScope, aArray, aCount, aFillValue);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode Conv1D(Scope aScope, OpNode aArray0, OpNode aArray1)
        {
            uint lNewOpNodeID = CppCall.OpNode_Conv1D(aScope, aArray0, aArray1);

            return new OpNode(lNewOpNodeID, ref aScope);
        }

        public static OpNode HCat(Scope aScope, OpNode aArray0, OpNode aArray1)
        {
            uint lNewOpNodeID = CppCall.OpNode_HCat(aScope, aArray0, aArray1);

            return new OpNode(lNewOpNodeID, ref aScope);
        }
    };
}