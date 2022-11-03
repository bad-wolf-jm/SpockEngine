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

        public override int GetHashCode()
        {
            unchecked
            {
                return ((((x.GetHashCode()) * 397) ^ y.GetHashCode()) * 397) ^ z.GetHashCode();
            }
        }

        public bool Equals(vec3 rhs) => ((x.Equals(rhs.x) && y.Equals(rhs.y)) && z.Equals(rhs.z));
        
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is vec3 && Equals((vec3) obj);
        }
        public static bool operator==(vec3 lhs, vec3 rhs) => lhs.Equals(rhs);
        
        /// <summary>
        /// Returns true iff this does not equal rhs (component-wise).
        /// </summary>
        public static bool operator!=(vec3 lhs, vec3 rhs) => !lhs.Equals(rhs);


        public float Dot(vec3 rhs) => ((x * rhs.x + y * rhs.y) + z * rhs.z);
        public vec3 Cross(vec3 r) => new vec3(y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x);

        public float[] Values => new[] { x, y, z };
        public float Length => (float)System.Math.Sqrt(((x*x + y*y) + z*z));
        public float Norm => (float)System.Math.Sqrt(((x*x + y*y) + z*z));
        public float Norm1 => ((System.Math.Abs(x) + System.Math.Abs(y)) + System.Math.Abs(z));
        public float Norm2 => (float)System.Math.Sqrt(((x*x + y*y) + z*z));
        public vec3 Normalized => this / (float)Length;

        public static vec3 operator-(vec3 lhs, vec3 rhs) => new vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
        public static vec3 operator+(vec3 lhs, vec3 rhs) => new vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);

        public static vec3 operator*(vec3 lhs, float rhs) => new vec3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
        public static vec3 operator*(float lhs, vec3 rhs) => new vec3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
        
        public static vec3 operator/(vec3 lhs, float rhs) => new vec3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
        public static vec3 operator/(float lhs, vec3 rhs) => new vec3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
    }
}