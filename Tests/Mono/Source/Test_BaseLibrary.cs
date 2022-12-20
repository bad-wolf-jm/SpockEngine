using System;
using SpockEngine.Math;

namespace SEUnitTest
{
    public class VectorTests
    {
        public static vec3 Add(vec3 x, vec3 y) { return x + y; }
        public static vec3 Subtract(vec3 x, vec3 y) { return x - y; }
        public static vec3 Divide0(vec3 x, float y) { return x / y; }
        public static vec3 Divide1(float x, vec3 y) { return x / y; }
        public static vec3 Multiply0(vec3 x, float y) { return x * y; }
        public static vec3 Multiply1(float x, vec3 y) { return x * y; }
        public static float Dot(vec3 x, vec3 y) { return x.Dot(y); }
        public static vec3 Cross(vec3 x, vec3 y) { return x.Cross(y); }
        public static float Length(vec3 x) { return x.Length; }
        public static float Norm(vec3 x) { return x.Norm; }
        public static float Norm1(vec3 x) { return x.Norm1; }
        public static float Norm2(vec3 x) { return x.Norm2; }
        public static vec3 Normalized(vec3 x) { return x.Normalized; }

    }
}