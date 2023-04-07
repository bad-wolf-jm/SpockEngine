using System;
using SpockEngine;
using SpockEngine.Math;

namespace SEUnitTest
{
    public class TensorOpsTest
    {
        public static sTensorShape CreateTensorShape(uint dummy)
        {
            uint[,] aShape = { { 2, 3, 4 }, { 5, 6, 7 }, { 8, 9, 10 }, { 2, 3, 2 } };
            uint aElementSize = 8;

            return new sTensorShape(aShape, aElementSize);
        }

        public static sTensorShape CreateRank1TensorShape(uint dummy)
        {
            uint[] aShape = { 2, 5, 3, 2 };
            uint aElementSize = 8;

            return new sTensorShape(aShape, aElementSize);
        }

        public static Scope CreateScope(uint aMemorySize)
        {
            return new Scope(aMemorySize);
        }

        public static OpNode CreateConstantMultiTensor(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            uint[,] aShape = { { 2, 3, 4 }, { 4, 2, 3 }, { 2, 4, 3 }, { 4, 3, 2 } };
            sTensorShape lShape = new sTensorShape(aShape, sizeof(float));

            return TensorOps.MultiTensorValue(lScope, new sConstantValueInitializerComponent<float>(1234.5f), lShape);
        }

        public static OpNode CreateLayeredConstantMultiTensor(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            uint[,] aShape = { { 2, 3, 4 }, { 4, 2, 3 }, { 2, 4, 3 }, { 4, 3, 2 } };
            sTensorShape lShape = new sTensorShape(aShape, sizeof(float));

            float[] lValues = { 1.2f, 3.4f, 4.5f, 6.7f };
            return TensorOps.MultiTensorValue(lScope, new sVectorInitializerComponent<float>(ref lValues), lShape);
        }

        public static OpNode CreateDataMultiTensor(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            uint[,] aShape = { { 2, 3 }, { 4, 2 } };
            sTensorShape lShape = new sTensorShape(aShape, sizeof(float));

            float[] lValues = { 1.2f, 3.4f, 4.5f, 6.7f, 8.9f, 10.11f, 1.2f, 3.4f, 4.5f, 6.7f, 8.9f, 10.11f, 12.13f, 14.15f };
            return TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref lValues), lShape);
        }

        public static OpNode CreateRandomUniformMultiTensor(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            uint[,] aShape = { { 2, 3 }, { 4, 2 } };
            sTensorShape lShape = new sTensorShape(aShape, sizeof(float));

            return TensorOps.MultiTensorValue(lScope, new sRandomUniformInitializerComponent<float>(), lShape);
        }

        public static OpNode CreateRandomNormalMultiTensor(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            uint[,] aShape = { { 2, 3 }, { 4, 2 } };
            sTensorShape lShape = new sTensorShape(aShape, sizeof(float));

            return TensorOps.MultiTensorValue(lScope, new sRandomNormalInitializerComponent<float>(1.234f, 2.345f), lShape);
        }

        public static OpNode CreateScalarValue(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            return TensorOps.ConstantScalarValue(lScope, 123.456f);
        }

        public static OpNode CreateScalarVector(ulong aScopeHandle)
        {
            Scope lScope = new Scope(aScopeHandle, false);

            float[] aValues = { 1.2f, 3.4f, 5.6f, 7.8f, 9.0f };

            return TensorOps.ScalarVectorValue(lScope, aValues);
        }

        public static OpNode TestAdd(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Add(lScope, lNode0, lNode1);
        }

        public static OpNode TestMultiply(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Multiply(lScope, lNode0, lNode1);
        }

        public static OpNode TestSubtract(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Subtract(lScope, lNode0, lNode1);
        }

        public static OpNode TestDivide(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Divide(lScope, lNode0, lNode1);
        }

        public static OpNode TestAnd(ulong aScopeHandle, ulong aShape, bool[] aData0, bool[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aData1), lShape);

            return TensorOps.And(lScope, lNode0, lNode1);
        }

        public static OpNode TestOr(ulong aScopeHandle, ulong aShape, bool[] aData0, bool[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aData1), lShape);

            return TensorOps.Or(lScope, lNode0, lNode1);
        }

        public static OpNode TestNot(ulong aScopeHandle, ulong aShape, bool[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aData0), lShape);

            return TensorOps.Not(lScope, lNode0);
        }

        public static OpNode TestBitwiseAnd(ulong aScopeHandle, ulong aShape, uint[] aData0, uint[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<uint>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<uint>(ref aData1), lShape);

            return TensorOps.BitwiseAnd(lScope, lNode0, lNode1);
        }

        public static OpNode TestBitwiseOr(ulong aScopeHandle, ulong aShape, uint[] aData0, uint[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<uint>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<uint>(ref aData1), lShape);

            return TensorOps.BitwiseOr(lScope, lNode0, lNode1);
        }

        public static OpNode TestBitwiseNot(ulong aScopeHandle, ulong aShape, uint[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<uint>(ref aData0), lShape);

            return TensorOps.BitwiseNot(lScope, lNode0);
        }

        public static OpNode TestInInterval(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1, float[] aData2)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);
            var lNode2 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData2), lShape);

            return TensorOps.InInterval(lScope, lNode0, lNode1, lNode2, false, false);
        }

        public static OpNode TestEqual(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Equal(lScope, lNode0, lNode1);
        }

        public static OpNode TestLessThan(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.LessThan(lScope, lNode0, lNode1);
        }

        public static OpNode TestLessThanOrEqual(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.LessThanOrEqual(lScope, lNode0, lNode1);
        }

        public static OpNode TestGreaterThan(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.GreaterThan(lScope, lNode0, lNode1);
        }

        public static OpNode TestGreaterThanOrEqual(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.GreaterThanOrEqual(lScope, lNode0, lNode1);
        }

        public static OpNode TestWhere(ulong aScopeHandle, ulong aShape, bool[] aCond, float[] aData0, float[] aData1)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lCond = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<bool>(ref aCond), lShape);
            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);

            return TensorOps.Where(lScope, lCond, lNode0, lNode1);
        }

        public static OpNode TestMix(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1, float[] aData2)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);
            var lNode2 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData2), lShape);

            return TensorOps.Mix(lScope, lNode0, lNode1, lNode2);
        }

        public static OpNode TestAffineTransform(ulong aScopeHandle, ulong aShape, float[] aData0, float[] aData1, float[] aData2)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);
            var lNode1 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData1), lShape);
            var lNode2 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData2), lShape);

            return TensorOps.Mix(lScope, lNode0, lNode1, lNode2);
        }

        public static OpNode TestCollapse(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Collapse(lScope, lNode0);
        }

        public static OpNode TestExpand(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Expand(lScope, lNode0);
        }

        public static OpNode TestFlatten(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Flatten(lScope, lNode0);
        }

        public static OpNode TestCountTrue(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.CountTrue(lScope, lNode0);
        }

        public static OpNode TestCountNonZero(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.CountNonZero(lScope, lNode0);
        }

        public static OpNode TestCountZero(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.CountZero(lScope, lNode0);
        }

        public static OpNode TestFloor(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Floor(lScope, lNode0);
        }

        public static OpNode TestCeil(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Ceil(lScope, lNode0);
        }

        public static OpNode TestAbs(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Abs(lScope, lNode0);
        }

        public static OpNode TestSqrt(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Sqrt(lScope, lNode0);
        }

        public static OpNode TestRound(ulong aScopeHandle, ulong aShape, float[] aData0)
        {
            Scope lScope = new Scope(aScopeHandle, false);
            sTensorShape lShape = new sTensorShape(aShape);

            var lNode0 = TensorOps.MultiTensorValue(lScope, new sDataInitializerComponent<float>(ref aData0), lShape);

            return TensorOps.Round(lScope, lNode0);
        }

    }
}
