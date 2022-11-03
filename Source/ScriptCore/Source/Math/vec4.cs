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

        public float Dot(vec4 rhs) => ((x * rhs.x + y * rhs.y) + z * rhs.z + w * rhs.w);

        public float[] Values => new[] { x, y, z, w };
        public float Length => (float)System.Math.Sqrt(((x*x + y*y) + z*z + w*w));
        public float Norm => (float)System.Math.Sqrt(((x*x + y*y) + z*z + w*w));
        public float Norm1 => ((System.Math.Abs(x) + System.Math.Abs(y)) + System.Math.Abs(z) + System.Math.Abs(w));
        public float Norm2 => (float)System.Math.Sqrt(((x*x + y*y) + z*z + w*w));
        public vec4 Normalized => this / (float)Length;

        public static vec4 operator-(vec4 lhs, vec4 rhs) => new vec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
        public static vec4 operator+(vec4 lhs, vec4 rhs) => new vec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);

        public static vec4 operator*(vec4 lhs, float rhs) => new vec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
        public static vec4 operator*(float lhs, vec4 rhs) => new vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
        
        public static vec4 operator/(vec4 lhs, float rhs) => new vec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
        public static vec4 operator/(float lhs, vec4 rhs) => new vec4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);


    }
}