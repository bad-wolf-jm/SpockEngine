using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct mat3
    {
        public float m00;
        public float m10;
        public float m20;

        public float m01;
        public float m11;
        public float m21;
        
        public float m02;
        public float m12;
        public float m22;

        public mat3(float aDiag)
        {
            m00 = aDiag; m01 = 0.0f;  m02 = 0.0f; 
            m10 = 0.0f;  m11 = aDiag; m12 = 0.0f; 
            m20 = 0.0f;  m21 = 0.0f;  m22 = aDiag;
        }
    }
}