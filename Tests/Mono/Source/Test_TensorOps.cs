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

            float[] lValues = {1.2f, 3.4f, 4.5f, 6.7f};
            return TensorOps.MultiTensorValue(lScope, new sVectorInitializerComponent<float>(ref lValues), lShape);
        }
    }
}
