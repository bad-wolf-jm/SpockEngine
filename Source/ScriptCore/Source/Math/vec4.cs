using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct vec4
    {
        public float x;
        public float y;
        public float z;
        public float w;

        public vec4(float aX, float aY, float aZ, float aW)
        {
            x = aX;
            y = aY; 
            z = aZ; 
            w = aW;  
        }

        public vec4(vec3 aProj, float aW)
        {
            x = aProj.x; 
            y = aProj.y;  
            z = aProj.z;  
            w = aW;  
        }

        public vec4(vec3 aProj)
        {
            x = aProj.x; 
            y = aProj.y;  
            z = aProj.z;  
            w = 0.0f;  
        }
    }
}