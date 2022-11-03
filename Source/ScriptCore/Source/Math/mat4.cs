using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct mat4
    {
        public float m00;
        public float m10;
        public float m20;
        public float m30;

        public float m01;
        public float m11;
        public float m21;
        public float m31;
        
        public float m02;
        public float m12;
        public float m22;
        public float m32;

        public float m03;
        public float m13;
        public float m23;
        public float m33;

        public mat4(float aDiag)
        {
            m00 = aDiag; m01 = 0.0f;  m02 = 0.0f;  m03 = 0.0f; 
            m10 = 0.0f;  m11 = aDiag; m12 = 0.0f;  m13 = 0.0f; 
            m20 = 0.0f;  m21 = 0.0f;  m22 = aDiag; m23 = 0.0f; 
            m30 = 0.0f;  m31 = 0.0f;  m32 = 0.0f;  m33 = aDiag; 
        }

        public mat4(mat3 aOther)
        {
            m00 = aOther.m00; m01 = aOther.m01; m02 = aOther.m02; m03 = 0.0f; 
            m10 = aOther.m10; m11 = aOther.m11; m12 = aOther.m12; m13 = 0.0f; 
            m20 = aOther.m20; m21 = aOther.m21; m22 = aOther.m22; m23 = 0.0f; 
            m30 = 0.0f;       m31 = 0.0f;       m32 = 0.0f;       m33 = 1.0f; 
        }

    }
}