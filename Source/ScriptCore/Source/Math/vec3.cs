using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct vec3
    {
        public float x;
        public float y;
        public float z;

        public vec3(float aX, float aY, float aZ)
        {
            x = aX;
            y = aY; 
            z = aZ; 
        }

        public vec3(vec4 aProj)
        {
            x = aProj.x;
            y = aProj.y; 
            z = aProj.z; 
        }
    }
}