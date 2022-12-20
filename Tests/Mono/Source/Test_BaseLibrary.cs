using System;
using SpockEngine.Math;

namespace SEUnitTest
{
    public class Vector3Tests
    {
        public static vec3 Constructor(float x, float y, float z) { return new vec3(x, y, z); }

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

    public class Vector4Tests
    {
        public static vec4 Constructor0(float x, float y, float z, float w) { return new vec4(x, y, z, w); }
        public static vec4 Constructor1(vec3 xyz, float w) { return new vec4(xyz, w); }
        public static vec4 Constructor2(vec3 xyz) { return new vec4(xyz); }
        public static vec3 Projection(vec4 v) { return new vec3(v); }
        public static vec4 Add(vec4 x, vec4 y) { return x + y; }
        public static vec4 Subtract(vec4 x, vec4 y) { return x - y; }
        public static vec4 Divide0(vec4 x, float y) { return x / y; }
        public static vec4 Divide1(float x, vec4 y) { return x / y; }
        public static vec4 Multiply0(vec4 x, float y) { return x * y; }
        public static vec4 Multiply1(float x, vec4 y) { return x * y; }
        public static float Dot(vec4 x, vec4 y) { return x.Dot(y); }
        public static float Length(vec4 x) { return x.Length; }
        public static float Norm(vec4 x) { return x.Norm; }
        public static float Norm1(vec4 x) { return x.Norm1; }
        public static float Norm2(vec4 x) { return x.Norm2; }
        public static vec4 Normalized(vec4 x) { return x.Normalized; }
    }

}