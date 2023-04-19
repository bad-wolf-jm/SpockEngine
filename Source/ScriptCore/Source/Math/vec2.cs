using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct vec2
    {
        public float x;
        public float y;

        public static implicit operator vec2(float[] values)
        {
            return new vec2(values[0], values[1]);
        }

        public static implicit operator vec2(double[] values)
        {
            return new vec2((float)values[0], (float)values[1]);
        }

        public vec2(float aX, float aY)
        {
            x = aX;
            y = aY;
        }

        public vec2(vec4 aProj)
        {
            x = aProj.x;
            y = aProj.y;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (((x.GetHashCode()) * 397) ^ y.GetHashCode());
            }
        }

        public bool Equals(vec2 rhs) => (x.Equals(rhs.x) && y.Equals(rhs.y));

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is vec2 && Equals((vec2)obj);
        }

        public static bool operator ==(vec2 lhs, vec2 rhs) => lhs.Equals(rhs);

        /// <summary>
        /// Returns true iff this does not equal rhs (component-wise).
        /// </summary>
        public static bool operator !=(vec2 lhs, vec2 rhs) => !lhs.Equals(rhs);


        public float Dot(vec2 rhs) => (x * rhs.x + y * rhs.y);

        public float[] Values => new[] { x, y };

        public float Length => (float)System.Math.Sqrt(x * x + y * y);

        public float Norm => (float)System.Math.Sqrt(x * x + y * y);

        public float Norm1 => (System.Math.Abs(x) + System.Math.Abs(y));

        public float Norm2 => (float)System.Math.Sqrt((x * x + y * y));

        public vec2 Normalized => this / (float)Length;

        public static vec2 operator -(vec2 lhs, vec2 rhs) => new vec2(lhs.x - rhs.x, lhs.y - rhs.y);

        public static vec2 operator +(vec2 lhs, vec2 rhs) => new vec2(lhs.x + rhs.x, lhs.y + rhs.y);

        public static vec2 operator *(vec2 lhs, float rhs) => new vec2(lhs.x * rhs, lhs.y * rhs);

        public static vec2 operator *(float lhs, vec2 rhs) => new vec2(lhs * rhs.x, lhs * rhs.y);

        public static vec2 operator /(vec2 lhs, float rhs) => new vec2(lhs.x / rhs, lhs.y / rhs);

        public static vec2 operator /(float lhs, vec2 rhs) => new vec2(lhs / rhs.x, lhs / rhs.y);
    }
}