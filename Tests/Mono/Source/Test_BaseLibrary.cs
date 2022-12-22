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

    public class Matrix3Tests
    {
        public static mat3 Constructor0(float x) { return new mat3(x); }
        public static mat3 Constructor1(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
        {
            return new mat3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
        }
        public static mat3 Constructor2(mat4 x) { return new mat3(x); }

        public static vec3 Column0(mat3 aM) { return aM.mC0; }
        public static vec3 Column1(mat3 aM) { return aM.mC1; }
        public static vec3 Column2(mat3 aM) { return aM.mC2; }

        public static mat3 Transposed(mat3 aM) { return aM.Transposed; }
        public static float Determinant(mat3 aM) { return aM.Determinant(); }
        public static mat3 Inverse(mat3 aM) { return aM.Inverse(); }

        public static mat3 Divide0(mat3 x, float y) { return x / y; }
        public static mat3 Divide1(float x, mat3 y) { return x / y; }

        public static mat3 Multiply0(mat3 x, float y) { return x * y; }
        public static mat3 Multiply1(float x, mat3 y) { return x * y; }
        public static mat3 Multiply2(mat3 x, mat3 y) { return x * y; }
        public static vec3 Multiply3(mat3 x, vec3 y) { return x * y; }

        public static mat3 Add(mat3 x, mat3 y) { return x + y; }
        public static mat3 Subtract(mat3 x, mat3 y) { return x - y; }
    }

    public class Matrix4Tests
    {
        public static mat4 Constructor0(float x) { return new mat4(x); }
        public static mat4 Constructor1(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13,
                                        float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33)
        {
            return new mat4(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
        }
        public static mat4 Constructor2(mat3 x) { return new mat4(x); }

        public static vec4 Column0(mat4 aM) { return aM.mC0; }
        public static vec4 Column1(mat4 aM) { return aM.mC1; }
        public static vec4 Column2(mat4 aM) { return aM.mC2; }
        public static vec4 Column3(mat4 aM) { return aM.mC3; }

        public static mat4 Transposed(mat4 aM) { return aM.Transposed; }
        public static float Determinant(mat4 aM) { return aM.Determinant(); }
        public static mat4 Inverse(mat4 aM) { return aM.Inverse(); }

        public static mat4 Divide0(mat4 x, float y) { return x / y; }
        public static mat4 Divide1(float x, mat4 y) { return x / y; }

        public static mat4 Multiply0(mat4 x, float y) { return x * y; }
        public static mat4 Multiply1(float x, mat4 y) { return x * y; }
        public static mat4 Multiply2(mat4 x, mat4 y) { return x * y; }
        public static vec4 Multiply3(mat4 x, vec4 y) { return x * y; }

        public static mat4 Add(mat4 x, mat4 y) { return x + y; }
        public static mat4 Subtract(mat4 x, mat4 y) { return x - y; }
    }

}