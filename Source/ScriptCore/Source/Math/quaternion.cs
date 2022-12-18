// using System;
// using System.Collections;
// using System.Collections.Generic;
// using System.Globalization;
using System.Runtime.InteropServices;
using SpockEngine;
// using System.Runtime.Serialization;
// using System.Numerics;
// using System.Linq;
// using GlmSharp.Swizzle;

// ReSharper disable InconsistentNaming

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct quat
    {
        public float x;
        public float y;
        public float z;
        public float w;


        public quat(float aX, float aY, float aZ, float aW) { x = aX; y = aY; z = aZ; w = aW; }

        public quat(float aV) { x = aV; y = aV; z = aV; w = aV; }

        public quat(quat aQ) { x = aQ.x; y = aQ.y; z = aQ.z; w = aQ.w; }

        public quat(vec3 aV, float aS) { x = aV.x; y = aV.y; z = aV.z; w = aS; }

        public quat(vec3 aU, vec3 aV)
        {
            var lW = aU.Cross(aV);
            var dot = aU.Dot(aV);
            var lQ = new quat(lW.x, lW.y, lW.z, 1.0f + dot).Normalized;

            x = lQ.x;
            y = lQ.y;
            z = lQ.z;
            w = lQ.w;
        }

        /// <summary>
        /// Create a quaternion from two normalized axis (http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors)
        /// </summary>
        public quat(vec3 aEulerAngle)
        {
            var lHalfAngle = aEulerAngle / 2.0f;
            var lCos = new vec3(Functions.cos(lHalfAngle.x), Functions.cos(lHalfAngle.y), Functions.cos(lHalfAngle.z));
            var lSin = new vec3(Functions.sin(lHalfAngle.x), Functions.sin(lHalfAngle.y), Functions.sin(lHalfAngle.z));

            x = lSin.x * lCos.y * lCos.z - lCos.x * lSin.y * lSin.z;
            y = lCos.x * lSin.y * lCos.z + lSin.x * lCos.y * lSin.z;
            z = lCos.x * lCos.y * lSin.z - lSin.x * lSin.y * lCos.z;
            w = lCos.x * lCos.y * lCos.z + lSin.x * lSin.y * lSin.z;
        }

        public quat(mat3 aMatrix) : this(FromMat3(aMatrix)) { }

        /// <summary>
        /// Creates a quaternion from the rotational part of a mat4.
        /// </summary>
        public quat(mat4 aMatrix) : this(FromMat4(aMatrix)) { }

        public static explicit operator vec4(quat aQ) => new vec4((float)aQ.x, (float)aQ.y, (float)aQ.z, (float)aQ.w);
        public static explicit operator quat(mat3 aMatrix) => FromMat3(aMatrix);
        public static explicit operator quat(mat4 aMatrix) => FromMat4(aMatrix);

        // #region Indexer

        // /// <summary>
        // /// Gets/Sets a specific indexed component (a bit slower than direct access).
        // /// </summary>
        // public float this[int index]
        // {
        //     get
        //     {
        //         switch (index)
        //         {
        //             case 0: return x;
        //             case 1: return y;
        //             case 2: return z;
        //             case 3: return w;
        //             default: throw new ArgumentOutOfRangeException("index");
        //         }
        //     }
        //     set
        //     {
        //         switch (index)
        //         {
        //             case 0: x = value; break;
        //             case 1: y = value; break;
        //             case 2: z = value; break;
        //             case 3: w = value; break;
        //             default: throw new ArgumentOutOfRangeException("index");
        //         }
        //     }
        // }

        // #endregion


        // #region Properties

        /// <summary>
        /// Returns an array with all values
        /// </summary>
        public float[] Values => new[] { x, y, z, w };

        // /// <summary>
        // /// Returns the number of components (4).
        // /// </summary>
        // public int Count => 4;

        /// <summary>
        /// Returns the euclidean length of this quaternion.
        /// </summary>
        public float Length => (float)System.Math.Sqrt(((x * x + y * y) + (z * z + w * w)));

        /// <summary>
        /// Returns the squared euclidean length of this quaternion.
        /// </summary>
        public float LengthSqr => ((x * x + y * y) + (z * z + w * w));

        /// <summary>
        /// Returns a copy of this quaternion with length one (undefined if this has zero length).
        /// </summary>
        public quat Normalized => this / (float)Length;

        // /// <summary>
        // /// Returns a copy of this quaternion with length one (returns zero if length is zero).
        // /// </summary>
        // public quat NormalizedSafe => this == Zero ? Identity : this / (float)Length;

        /// <summary>
        /// Returns the represented angle of this quaternion.
        /// </summary>
        public double Angle => Functions.acos(w) * 2.0;

        /// <summary>
        /// Returns the represented axis of this quaternion.
        /// </summary>
        public vec3 Axis
        {
            get
            {
                var s1 = 1 - w * w;
                if (s1 < 0) return new vec3(0.0f, 0.0f, 1.0f);
                var s2 = 1 / System.Math.Sqrt(s1);
                return new vec3((float)(x * s2), (float)(y * s2), (float)(z * s2));
            }
        }

        // /// <summary>
        // /// Returns the represented yaw angle of this quaternion.
        // /// </summary>
        // public double Yaw => System.Math.Asin(-2.0 * (x * z - w * y));

        // /// <summary>
        // /// Returns the represented pitch angle of this quaternion.
        // /// </summary>
        // public double Pitch => System.Math.Atan2(2.0 * (y * z + w * x), (w * w - x * x - y * y + z * z));

        // /// <summary>
        // /// Returns the represented roll angle of this quaternion.
        // /// </summary>
        // public double Roll => System.Math.Atan2(2.0 * (x * y + w * z), (w * w + x * x - y * y - z * z));

        /// <summary>
        /// Returns the represented euler angles (pitch, yaw, roll) of this quaternion.
        /// </summary>
        public vec3 EulerAngles => new vec3(
            (float)System.Math.Atan2(2.0 * (y * z + w * x), (w * w - x * x - y * y + z * z)),
            (float)System.Math.Asin(-2.0 * (x * z - w * y)),
            (float)System.Math.Atan2(2.0 * (x * y + w * z), (w * w + x * x - y * y - z * z))
        );

        /// <summary>
        /// Creates a mat3 that realizes the rotation of this quaternion
        /// </summary>
        public mat3 ToMat3 => new mat3(
            1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
            2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
            2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
        );

        public mat4 ToMat4 => new mat4(ToMat3);
        public quat Conjugate => new quat(-x, -y, -z, w);
        public quat Inverse => Conjugate / LengthSqr;

        // #endregion


        // #region Static Properties

        // /// <summary>
        // /// Predefined all-zero quaternion
        // /// </summary>
        // public static quat Zero { get; } = new quat(0f, 0f, 0f, 0f);

        // /// <summary>
        // /// Predefined all-ones quaternion
        // /// </summary>
        // public static quat Ones { get; } = new quat(1f, 1f, 1f, 1f);

        // /// <summary>
        // /// Predefined identity quaternion
        // /// </summary>
        // public static quat Identity { get; } = new quat(0f, 0f, 0f, 1f);

        // /// <summary>
        // /// Predefined unit-X quaternion
        // /// </summary>
        // public static quat UnitX { get; } = new quat(1f, 0f, 0f, 0f);

        // /// <summary>
        // /// Predefined unit-Y quaternion
        // /// </summary>
        // public static quat UnitY { get; } = new quat(0f, 1f, 0f, 0f);

        // /// <summary>
        // /// Predefined unit-Z quaternion
        // /// </summary>
        // public static quat UnitZ { get; } = new quat(0f, 0f, 1f, 0f);

        // /// <summary>
        // /// Predefined unit-W quaternion
        // /// </summary>
        // public static quat UnitW { get; } = new quat(0f, 0f, 0f, 1f);

        // /// <summary>
        // /// Predefined all-MaxValue quaternion
        // /// </summary>
        // public static quat MaxValue { get; } = new quat(float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue);

        // /// <summary>
        // /// Predefined all-MinValue quaternion
        // /// </summary>
        // public static quat MinValue { get; } = new quat(float.MinValue, float.MinValue, float.MinValue, float.MinValue);

        // /// <summary>
        // /// Predefined all-Epsilon quaternion
        // /// </summary>
        // public static quat Epsilon { get; } = new quat(float.Epsilon, float.Epsilon, float.Epsilon, float.Epsilon);

        // /// <summary>
        // /// Predefined all-NaN quaternion
        // /// </summary>
        // public static quat NaN { get; } = new quat(float.NaN, float.NaN, float.NaN, float.NaN);

        // /// <summary>
        // /// Predefined all-NegativeInfinity quaternion
        // /// </summary>
        // public static quat NegativeInfinity { get; } = new quat(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        // /// <summary>
        // /// Predefined all-PositiveInfinity quaternion
        // /// </summary>
        // public static quat PositiveInfinity { get; } = new quat(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);

        // #endregion


        // #region Operators

        public static bool operator ==(quat aLeft, quat aRight) => aLeft.Equals(aRight);
        public static bool operator !=(quat aLeft, quat aRight) => !aLeft.Equals(aRight);

        public static quat operator *(quat p, quat q) => new quat(p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y, p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z, p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x, p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z);

        public static vec3 operator *(quat q, vec3 v)
        {
            var qv = new vec3(q.x, q.y, q.z);
            var uv = qv.Cross(v);
            var uuv = qv.Cross(uv);
            return v + ((uv * q.w) + uuv) * 2;
        }

        public static vec4 operator *(quat q, vec4 v) => new vec4(q * new vec3(v), v.w);
        public static vec3 operator *(vec3 v, quat q) => q.Inverse * v;
        public static vec4 operator *(vec4 v, quat q) => q.Inverse * v;



        // #region Functions

        // /// <summary>
        // /// Returns an enumerator that iterates through all components.
        // /// </summary>
        // public IEnumerator<float> GetEnumerator()
        // {
        //     yield return x;
        //     yield return y;
        //     yield return z;
        //     yield return w;
        // }

        // /// <summary>
        // /// Returns an enumerator that iterates through all components.
        // /// </summary>
        // IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        // /// <summary>
        // /// Returns a string representation of this quaternion using ', ' as a seperator.
        // /// </summary>
        // public override string ToString() => ToString(", ");

        // /// <summary>
        // /// Returns a string representation of this quaternion using a provided seperator.
        // /// </summary>
        // public string ToString(string sep) => ((x + sep + y) + sep + (z + sep + w));

        // /// <summary>
        // /// Returns a string representation of this quaternion using a provided seperator and a format provider for each component.
        // /// </summary>
        // public string ToString(string sep, IFormatProvider provider) => ((x.ToString(provider) + sep + y.ToString(provider)) + sep + (z.ToString(provider) + sep + w.ToString(provider)));

        // /// <summary>
        // /// Returns a string representation of this quaternion using a provided seperator and a format for each component.
        // /// </summary>
        // public string ToString(string sep, string format) => ((x.ToString(format) + sep + y.ToString(format)) + sep + (z.ToString(format) + sep + w.ToString(format)));

        // /// <summary>
        // /// Returns a string representation of this quaternion using a provided seperator and a format and format provider for each component.
        // /// </summary>
        // public string ToString(string sep, string format, IFormatProvider provider) => ((x.ToString(format, provider) + sep + y.ToString(format, provider)) + sep + (z.ToString(format, provider) + sep + w.ToString(format, provider)));

        /// <summary>
        /// Returns true iff this equals aRight component-wise.
        /// </summary>
        public bool Equals(quat aRight) => ((x.Equals(aRight.x) && y.Equals(aRight.y)) && (z.Equals(aRight.z) && w.Equals(aRight.w)));

        /// <summary>
        /// Returns true iff this equals aRight type- and component-wise.
        /// </summary>
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is quat && Equals((quat)obj);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        public override int GetHashCode()
        {
            unchecked
            {
                return ((((((x.GetHashCode()) * 397) ^ y.GetHashCode()) * 397) ^ z.GetHashCode()) * 397) ^ w.GetHashCode();
            }
        }

        /// <summary>
        /// Rotates this quaternion from an axis and an angle (in radians).
        /// </summary>
        public quat Rotated(float angle, vec3 v) => this * FromAxisAngle(angle, v);

        // #endregion


        // #region Static Functions

        // /// <summary>
        // /// Converts the string representation of the quaternion into a quaternion representation (using ', ' as a separator).
        // /// </summary>
        // public static quat Parse(string s) => Parse(s, ", ");

        // /// <summary>
        // /// Converts the string representation of the quaternion into a quaternion representation (using a designated separator).
        // /// </summary>
        // public static quat Parse(string s, string sep)
        // {
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) throw new FormatException("input has not exactly 4 parts");
        //     return new quat(float.Parse(kvp[0].Trim()), float.Parse(kvp[1].Trim()), float.Parse(kvp[2].Trim()), float.Parse(kvp[3].Trim()));
        // }

        // /// <summary>
        // /// Converts the string representation of the quaternion into a quaternion representation (using a designated separator and a type provider).
        // /// </summary>
        // public static quat Parse(string s, string sep, IFormatProvider provider)
        // {
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) throw new FormatException("input has not exactly 4 parts");
        //     return new quat(float.Parse(kvp[0].Trim(), provider), float.Parse(kvp[1].Trim(), provider), float.Parse(kvp[2].Trim(), provider), float.Parse(kvp[3].Trim(), provider));
        // }

        // /// <summary>
        // /// Converts the string representation of the quaternion into a quaternion representation (using a designated separator and a number style).
        // /// </summary>
        // public static quat Parse(string s, string sep, NumberStyles style)
        // {
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) throw new FormatException("input has not exactly 4 parts");
        //     return new quat(float.Parse(kvp[0].Trim(), style), float.Parse(kvp[1].Trim(), style), float.Parse(kvp[2].Trim(), style), float.Parse(kvp[3].Trim(), style));
        // }

        // /// <summary>
        // /// Converts the string representation of the quaternion into a quaternion representation (using a designated separator and a number style and a format provider).
        // /// </summary>
        // public static quat Parse(string s, string sep, NumberStyles style, IFormatProvider provider)
        // {
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) throw new FormatException("input has not exactly 4 parts");
        //     return new quat(float.Parse(kvp[0].Trim(), style, provider), float.Parse(kvp[1].Trim(), style, provider), float.Parse(kvp[2].Trim(), style, provider), float.Parse(kvp[3].Trim(), style, provider));
        // }

        // // /// <summary>
        // /// Tries to convert the string representation of the quaternion into a quaternion representation (using ', ' as a separator), returns false if string was invalid.
        // /// </summary>
        // public static bool TryParse(string s, out quat result) => TryParse(s, ", ", out result);

        // /// <summary>
        // /// Tries to convert the string representation of the quaternion into a quaternion representation (using a designated separator), returns false if string was invalid.
        // /// </summary>
        // public static bool TryParse(string s, string sep, out quat result)
        // {
        //     result = Zero;
        //     if (string.IsNullOrEmpty(s)) return false;
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) return false;
        //     float x = 0f, y = 0f, z = 0f, w = 0f;
        //     var ok = ((float.TryParse(kvp[0].Trim(), out x) && float.TryParse(kvp[1].Trim(), out y)) && (float.TryParse(kvp[2].Trim(), out z) && float.TryParse(kvp[3].Trim(), out w)));
        //     result = ok ? new quat(x, y, z, w) : Zero;
        //     return ok;
        // }

        // /// <summary>
        // /// Tries to convert the string representation of the quaternion into a quaternion representation (using a designated separator and a number style and a format provider), returns false if string was invalid.
        // /// </summary>
        // public static bool TryParse(string s, string sep, NumberStyles style, IFormatProvider provider, out quat result)
        // {
        //     result = Zero;
        //     if (string.IsNullOrEmpty(s)) return false;
        //     var kvp = s.Split(new[] { sep }, StringSplitOptions.None);
        //     if (kvp.Length != 4) return false;
        //     float x = 0f, y = 0f, z = 0f, w = 0f;
        //     var ok = ((float.TryParse(kvp[0].Trim(), style, provider, out x) && float.TryParse(kvp[1].Trim(), style, provider, out y)) && (float.TryParse(kvp[2].Trim(), style, provider, out z) && float.TryParse(kvp[3].Trim(), style, provider, out w)));
        //     result = ok ? new quat(x, y, z, w) : Zero;
        //     return ok;
        // }

        /// <summary>
        /// Returns the inner product (dot product, scalar product) of the two quaternions.
        /// </summary>
        public static float Dot(quat aLeft, quat aRight) => ((aLeft.x * aRight.x + aLeft.y * aRight.y) + (aLeft.z * aRight.z + aLeft.w * aRight.w));

        /// <summary>
        /// Creates a quaternion from an axis and an angle (in radians).
        /// </summary>
        public static quat FromAxisAngle(float angle, vec3 v)
        {
            var s = Functions.sin(angle * 0.5f);
            var c = Functions.cos(angle * 0.5f);
            return new quat((float)(v.x * s), (float)(v.y * s), (float)(v.z * s), (float)c);
        }

        /// <summary>
        /// Creates a quaternion from the rotational part of a mat3.
        /// </summary>
        public static quat FromMat3(mat3 m)
        {
            float lFourXSquaredMinus1 = m.m00 - m.m11 - m.m22;
            float lFourYSquaredMinus1 = m.m11 - m.m00 - m.m22;
            float lFourZSquaredMinus1 = m.m22 - m.m00 - m.m11;
            float lFourWSquaredMinus1 = m.m00 + m.m11 + m.m22;
            float lBiggestIndex = 0;
            float fourBiggestSquaredMinus1 = lFourWSquaredMinus1;
            if (lFourXSquaredMinus1 > fourBiggestSquaredMinus1)
            {
                fourBiggestSquaredMinus1 = lFourXSquaredMinus1;
                lBiggestIndex = 1;
            }
            if (lFourYSquaredMinus1 > fourBiggestSquaredMinus1)
            {
                fourBiggestSquaredMinus1 = lFourYSquaredMinus1;
                lBiggestIndex = 2;
            }
            if (lFourZSquaredMinus1 > fourBiggestSquaredMinus1)
            {
                fourBiggestSquaredMinus1 = lFourZSquaredMinus1;
                lBiggestIndex = 3;
            }
            float lBiggestValue = (float)System.Math.Sqrt(fourBiggestSquaredMinus1 + 1.0f) * 0.5f;
            float mult = 0.25f / lBiggestValue;
            switch (lBiggestIndex)
            {
                case 0: return new quat(lBiggestValue, (m.m21 + m.m12) * mult, (m.m02 + m.m20) * mult, (m.m10 - m.m01) * mult);
                case 1: return new quat((m.m21 + m.m12) * mult, lBiggestValue, (m.m10 + m.m01) * mult, (m.m02 - m.m20) * mult);
                case 2: return new quat((m.m02 + m.m20) * mult, (m.m10 + m.m01) * mult, lBiggestValue, (m.m21 - m.m12) * mult);
                case 3: return new quat((m.m10 - m.m01) * mult, (m.m02 - m.m20) * mult, (m.m21 - m.m12) * mult, lBiggestValue);
                default: return new quat(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        /// <summary>
        /// Creates a quaternion from the rotational part of a mat4.
        /// </summary>
        public static quat FromMat4(mat4 m) => FromMat3(new mat3(m));

        /// <summary>
        /// Returns the cross product between two quaternions.
        /// </summary>
        public static quat Cross(quat q1, quat q2) => new quat(
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,
            q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x,
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        );

        /// <summary>
        /// Calculates a proper spherical interpolation between two quaternions (only works for normalized quaternions).
        /// </summary>
        public static quat Mix(quat x, quat y, float a)
        {
            var cosTheta = Dot(x, y);
            if (cosTheta > 1 - float.Epsilon)
                return Lerp(x, y, a);
            else
            {
                var angle = Functions.acos(cosTheta);
                return (quat)(((float)(Functions.sin((1 - a) * angle)) * x + (Functions.sin(a * angle)) * y) / Functions.sin(angle));
            }
        }

        /// <summary>
        /// Calculates a proper spherical interpolation between two quaternions (only works for normalized quaternions).
        /// </summary>
        public static quat SLerp(quat x, quat y, float a)
        {
            var z = y;
            var cosTheta = Dot(x, y);
            if (cosTheta < 0) { z = -y; cosTheta = -cosTheta; }
            if (cosTheta > 1 - float.Epsilon)
                return Lerp(x, z, a);
            else
            {
                var angle = Functions.acos(cosTheta);
                return (quat)(((Functions.sin((1 - a) * angle)) * x + (Functions.sin(a * angle)) * z) / Functions.sin(angle));
            }
        }

        /// <summary>
        /// Applies squad interpolation of these quaternions
        /// </summary>
        public static quat Squad(quat q1, quat q2, quat s1, quat s2, float h) => Mix(Mix(q1, q2, h), Mix(s1, s2, h), 2 * (1 - h) * h);

        // #endregion


        // #region Component-Wise Static Functions

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of IsInfinity (float.IsInfinity(v)).
        // /// </summary>
        // public static bvec4 IsInfinity(quat v) => new bvec4(float.IsInfinity(v.x), float.IsInfinity(v.y), float.IsInfinity(v.z), float.IsInfinity(v.w));

        // /// <summary>
        // /// Returns a bvec from the application of IsInfinity (float.IsInfinity(v)).
        // /// </summary>
        // public static bvec4 IsInfinity(float v) => new bvec4(float.IsInfinity(v));

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of IsFinite (!float.IsNaN(v) &amp;&amp; !float.IsInfinity(v)).
        // /// </summary>
        // public static bvec4 IsFinite(quat v) => new bvec4(!float.IsNaN(v.x) && !float.IsInfinity(v.x), !float.IsNaN(v.y) && !float.IsInfinity(v.y), !float.IsNaN(v.z) && !float.IsInfinity(v.z), !float.IsNaN(v.w) && !float.IsInfinity(v.w));

        // /// <summary>
        // /// Returns a bvec from the application of IsFinite (!float.IsNaN(v) &amp;&amp; !float.IsInfinity(v)).
        // /// </summary>
        // public static bvec4 IsFinite(float v) => new bvec4(!float.IsNaN(v) && !float.IsInfinity(v));

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of IsNaN (float.IsNaN(v)).
        // /// </summary>
        // public static bvec4 IsNaN(quat v) => new bvec4(float.IsNaN(v.x), float.IsNaN(v.y), float.IsNaN(v.z), float.IsNaN(v.w));

        // /// <summary>
        // /// Returns a bvec from the application of IsNaN (float.IsNaN(v)).
        // /// </summary>
        // public static bvec4 IsNaN(float v) => new bvec4(float.IsNaN(v));

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of IsNegativeInfinity (float.IsNegativeInfinity(v)).
        // /// </summary>
        // public static bvec4 IsNegativeInfinity(quat v) => new bvec4(float.IsNegativeInfinity(v.x), float.IsNegativeInfinity(v.y), float.IsNegativeInfinity(v.z), float.IsNegativeInfinity(v.w));

        // /// <summary>
        // /// Returns a bvec from the application of IsNegativeInfinity (float.IsNegativeInfinity(v)).
        // /// </summary>
        // public static bvec4 IsNegativeInfinity(float v) => new bvec4(float.IsNegativeInfinity(v));

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of IsPositiveInfinity (float.IsPositiveInfinity(v)).
        // /// </summary>
        // public static bvec4 IsPositiveInfinity(quat v) => new bvec4(float.IsPositiveInfinity(v.x), float.IsPositiveInfinity(v.y), float.IsPositiveInfinity(v.z), float.IsPositiveInfinity(v.w));

        // /// <summary>
        // /// Returns a bvec from the application of IsPositiveInfinity (float.IsPositiveInfinity(v)).
        // /// </summary>
        // public static bvec4 IsPositiveInfinity(float v) => new bvec4(float.IsPositiveInfinity(v));

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of Equal (aLeft == aRight).
        // /// </summary>
        // public static bvec4 Equal(quat aLeft, quat aRight) => new bvec4(aLeft.x == aRight.x, aLeft.y == aRight.y, aLeft.z == aRight.z, aLeft.w == aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of Equal (aLeft == aRight).
        // /// </summary>
        // public static bvec4 Equal(quat aLeft, float aRight) => new bvec4(aLeft.x == aRight, aLeft.y == aRight, aLeft.z == aRight, aLeft.w == aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of Equal (aLeft == aRight).
        // /// </summary>
        // public static bvec4 Equal(float aLeft, quat aRight) => new bvec4(aLeft == aRight.x, aLeft == aRight.y, aLeft == aRight.z, aLeft == aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of Equal (aLeft == aRight).
        // /// </summary>
        // public static bvec4 Equal(float aLeft, float aRight) => new bvec4(aLeft == aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of NotEqual (aLeft != aRight).
        // /// </summary>
        // public static bvec4 NotEqual(quat aLeft, quat aRight) => new bvec4(aLeft.x != aRight.x, aLeft.y != aRight.y, aLeft.z != aRight.z, aLeft.w != aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of NotEqual (aLeft != aRight).
        // /// </summary>
        // public static bvec4 NotEqual(quat aLeft, float aRight) => new bvec4(aLeft.x != aRight, aLeft.y != aRight, aLeft.z != aRight, aLeft.w != aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of NotEqual (aLeft != aRight).
        // /// </summary>
        // public static bvec4 NotEqual(float aLeft, quat aRight) => new bvec4(aLeft != aRight.x, aLeft != aRight.y, aLeft != aRight.z, aLeft != aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of NotEqual (aLeft != aRight).
        // /// </summary>
        // public static bvec4 NotEqual(float aLeft, float aRight) => new bvec4(aLeft != aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThan (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 GreaterThan(quat aLeft, quat aRight) => new bvec4(aLeft.x > aRight.x, aLeft.y > aRight.y, aLeft.z > aRight.z, aLeft.w > aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThan (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 GreaterThan(quat aLeft, float aRight) => new bvec4(aLeft.x > aRight, aLeft.y > aRight, aLeft.z > aRight, aLeft.w > aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThan (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 GreaterThan(float aLeft, quat aRight) => new bvec4(aLeft > aRight.x, aLeft > aRight.y, aLeft > aRight.z, aLeft > aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of GreaterThan (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 GreaterThan(float aLeft, float aRight) => new bvec4(aLeft > aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThanEqual (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 GreaterThanEqual(quat aLeft, quat aRight) => new bvec4(aLeft.x >= aRight.x, aLeft.y >= aRight.y, aLeft.z >= aRight.z, aLeft.w >= aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThanEqual (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 GreaterThanEqual(quat aLeft, float aRight) => new bvec4(aLeft.x >= aRight, aLeft.y >= aRight, aLeft.z >= aRight, aLeft.w >= aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of GreaterThanEqual (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 GreaterThanEqual(float aLeft, quat aRight) => new bvec4(aLeft >= aRight.x, aLeft >= aRight.y, aLeft >= aRight.z, aLeft >= aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of GreaterThanEqual (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 GreaterThanEqual(float aLeft, float aRight) => new bvec4(aLeft >= aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThan (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 LesserThan(quat aLeft, quat aRight) => new bvec4(aLeft.x < aRight.x, aLeft.y < aRight.y, aLeft.z < aRight.z, aLeft.w < aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThan (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 LesserThan(quat aLeft, float aRight) => new bvec4(aLeft.x < aRight, aLeft.y < aRight, aLeft.z < aRight, aLeft.w < aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThan (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 LesserThan(float aLeft, quat aRight) => new bvec4(aLeft < aRight.x, aLeft < aRight.y, aLeft < aRight.z, aLeft < aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of LesserThan (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 LesserThan(float aLeft, float aRight) => new bvec4(aLeft < aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThanEqual (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 LesserThanEqual(quat aLeft, quat aRight) => new bvec4(aLeft.x <= aRight.x, aLeft.y <= aRight.y, aLeft.z <= aRight.z, aLeft.w <= aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThanEqual (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 LesserThanEqual(quat aLeft, float aRight) => new bvec4(aLeft.x <= aRight, aLeft.y <= aRight, aLeft.z <= aRight, aLeft.w <= aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of LesserThanEqual (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 LesserThanEqual(float aLeft, quat aRight) => new bvec4(aLeft <= aRight.x, aLeft <= aRight.y, aLeft <= aRight.z, aLeft <= aRight.w);

        // /// <summary>
        // /// Returns a bvec from the application of LesserThanEqual (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 LesserThanEqual(float aLeft, float aRight) => new bvec4(aLeft <= aRight);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(quat min, quat max, quat a) => new quat(min.x * (1-a.x) + max.x * a.x, min.y * (1-a.y) + max.y * a.y, min.z * (1-a.z) + max.z * a.z, min.w * (1-a.w) + max.w * a.w);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        public static quat Lerp(quat min, quat max, float a) => new quat(min.x * (1 - a) + max.x * a, min.y * (1 - a) + max.y * a, min.z * (1 - a) + max.z * a, min.w * (1 - a) + max.w * a);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(quat min, float max, quat a) => new quat(min.x * (1-a.x) + max * a.x, min.y * (1-a.y) + max * a.y, min.z * (1-a.z) + max * a.z, min.w * (1-a.w) + max * a.w);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(quat min, float max, float a) => new quat(min.x * (1-a) + max * a, min.y * (1-a) + max * a, min.z * (1-a) + max * a, min.w * (1-a) + max * a);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(float min, quat max, quat a) => new quat(min * (1-a.x) + max.x * a.x, min * (1-a.y) + max.y * a.y, min * (1-a.z) + max.z * a.z, min * (1-a.w) + max.w * a.w);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(float min, quat max, float a) => new quat(min * (1-a) + max.x * a, min * (1-a) + max.y * a, min * (1-a) + max.z * a, min * (1-a) + max.w * a);

        // /// <summary>
        // /// Returns a quat from component-wise application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(float min, float max, quat a) => new quat(min * (1-a.x) + max * a.x, min * (1-a.y) + max * a.y, min * (1-a.z) + max * a.z, min * (1-a.w) + max * a.w);

        // /// <summary>
        // /// Returns a quat from the application of Lerp (min * (1-a) + max * a).
        // /// </summary>
        // public static quat Lerp(float min, float max, float a) => new quat(min * (1-a) + max * a);

        // #endregion


        // #region Component-Wise Operator Overloads

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt; (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 operator<(quat aLeft, quat aRight) => new bvec4(aLeft.x < aRight.x, aLeft.y < aRight.y, aLeft.z < aRight.z, aLeft.w < aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt; (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 operator<(quat aLeft, float aRight) => new bvec4(aLeft.x < aRight, aLeft.y < aRight, aLeft.z < aRight, aLeft.w < aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt; (aLeft &lt; aRight).
        // /// </summary>
        // public static bvec4 operator<(float aLeft, quat aRight) => new bvec4(aLeft < aRight.x, aLeft < aRight.y, aLeft < aRight.z, aLeft < aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt;= (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 operator<=(quat aLeft, quat aRight) => new bvec4(aLeft.x <= aRight.x, aLeft.y <= aRight.y, aLeft.z <= aRight.z, aLeft.w <= aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt;= (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 operator<=(quat aLeft, float aRight) => new bvec4(aLeft.x <= aRight, aLeft.y <= aRight, aLeft.z <= aRight, aLeft.w <= aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&lt;= (aLeft &lt;= aRight).
        // /// </summary>
        // public static bvec4 operator<=(float aLeft, quat aRight) => new bvec4(aLeft <= aRight.x, aLeft <= aRight.y, aLeft <= aRight.z, aLeft <= aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt; (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 operator>(quat aLeft, quat aRight) => new bvec4(aLeft.x > aRight.x, aLeft.y > aRight.y, aLeft.z > aRight.z, aLeft.w > aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt; (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 operator>(quat aLeft, float aRight) => new bvec4(aLeft.x > aRight, aLeft.y > aRight, aLeft.z > aRight, aLeft.w > aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt; (aLeft &gt; aRight).
        // /// </summary>
        // public static bvec4 operator>(float aLeft, quat aRight) => new bvec4(aLeft > aRight.x, aLeft > aRight.y, aLeft > aRight.z, aLeft > aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt;= (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 operator>=(quat aLeft, quat aRight) => new bvec4(aLeft.x >= aRight.x, aLeft.y >= aRight.y, aLeft.z >= aRight.z, aLeft.w >= aRight.w);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt;= (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 operator>=(quat aLeft, float aRight) => new bvec4(aLeft.x >= aRight, aLeft.y >= aRight, aLeft.z >= aRight, aLeft.w >= aRight);

        // /// <summary>
        // /// Returns a bvec4 from component-wise application of operator&gt;= (aLeft &gt;= aRight).
        // /// </summary>
        // public static bvec4 operator>=(float aLeft, quat aRight) => new bvec4(aLeft >= aRight.x, aLeft >= aRight.y, aLeft >= aRight.z, aLeft >= aRight.w);

        public static quat operator +(quat v) => v;
        public static quat operator -(quat v) => new quat(-v.x, -v.y, -v.z, -v.w);
        public static quat operator +(quat aLeft, quat aRight) => new quat(aLeft.x + aRight.x, aLeft.y + aRight.y, aLeft.z + aRight.z, aLeft.w + aRight.w);
        public static quat operator -(quat aLeft, quat aRight) => new quat(aLeft.x - aRight.x, aLeft.y - aRight.y, aLeft.z - aRight.z, aLeft.w - aRight.w);
        public static quat operator *(quat aLeft, float aRight) => new quat(aLeft.x * aRight, aLeft.y * aRight, aLeft.z * aRight, aLeft.w * aRight);
        public static quat operator *(float aLeft, quat aRight) => new quat(aLeft * aRight.x, aLeft * aRight.y, aLeft * aRight.z, aLeft * aRight.w);
        public static quat operator /(quat aLeft, float aRight) => new quat(aLeft.x / aRight, aLeft.y / aRight, aLeft.z / aRight, aLeft.w / aRight);
        public static quat operator /(float aRight, quat aLeft) => new quat(aLeft.x / aRight, aLeft.y / aRight, aLeft.z / aRight, aLeft.w / aRight);

        // #endregion

    }
}