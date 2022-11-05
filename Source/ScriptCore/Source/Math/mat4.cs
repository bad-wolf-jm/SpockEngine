
using System.Runtime.InteropServices;

namespace SpockEngine.Math
{
    [StructLayout(LayoutKind.Sequential)]
    public struct mat4
    {
        public float m00;
        public float m10;
        public float m20;
        public float m30;

        public float m01;
        public float m11;
        public float m21;
        public float m31;
        
        public float m02;
        public float m12;
        public float m22;
        public float m32;

        public float m03;
        public float m13;
        public float m23;
        public float m33;

        public mat4(float aDiag)
        {
            m00 = aDiag; m01 = 0.0f;  m02 = 0.0f;  m03 = 0.0f; 
            m10 = 0.0f;  m11 = aDiag; m12 = 0.0f;  m13 = 0.0f; 
            m20 = 0.0f;  m21 = 0.0f;  m22 = aDiag; m23 = 0.0f; 
            m30 = 0.0f;  m31 = 0.0f;  m32 = 0.0f;  m33 = aDiag; 
        }


        public mat4(float a00, float a01, float a02, float a03, float a10, float a11, float a12, float a13, float a20, float a21, float a22, float a23, float a30, float a31, float a32, float a33)
        {
            m00 = a00; m01 = a01; m02 = a02; m03 = a03;
            m10 = a10; m11 = a11; m12 = a12; m13 = a13;
            m20 = a20; m21 = a21; m22 = a22; m23 = a23;
            m30 = a30; m31 = a31; m32 = a32; m33 = a33;
        }

        public mat4(mat3 aOther)
        {
            m00 = aOther.m00; m01 = aOther.m01; m02 = aOther.m02; m03 = 0.0f; 
            m10 = aOther.m10; m11 = aOther.m11; m12 = aOther.m12; m13 = 0.0f; 
            m20 = aOther.m20; m21 = aOther.m21; m22 = aOther.m22; m23 = 0.0f; 
            m30 = 0.0f;       m31 = 0.0f;       m32 = 0.0f;       m33 = 1.0f; 
        }

        public float[,] Values => new[,] { { m00, m10, m20, m30 }, { m01, m11, m21, m31 }, { m02, m12, m22, m32 }, { m03, m13, m23, m33 } };

        public vec4 mC0
        {
            get { return new vec4( m00, m10, m20, m30); }
            set { m00 = value.x; m10 = value.y; m20 = value.z; m30 = value.w; }
        }

        public vec4 mC1
        {
            get { return new vec4( m01, m11, m21, m31); }
            set { m01 = value.x; m11 = value.y; m21 = value.z; m31 = value.w; }
        }

        public vec4 mC2
        {
            get { return new vec4( m02, m12, m22, m32); }
            set { m02 = value.x; m12 = value.y; m22 = value.z; m32 = value.w; }
        }

        public vec4 mC3
        {
            get { return new vec4( m03, m13, m23, m33); }
            set { m03 = value.x; m13 = value.y; m23 = value.z; m33 = value.w; }
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return ((((((((((((((((((((((((((((((m00.GetHashCode()) * 397) ^ m01.GetHashCode()) * 397) ^ m02.GetHashCode()) * 397) ^ m03.GetHashCode()) * 397) ^ m10.GetHashCode()) * 397) ^ m11.GetHashCode()) * 397) ^ m12.GetHashCode()) * 397) ^ m13.GetHashCode()) * 397) ^ m20.GetHashCode()) * 397) ^ m21.GetHashCode()) * 397) ^ m22.GetHashCode()) * 397) ^ m23.GetHashCode()) * 397) ^ m30.GetHashCode()) * 397) ^ m31.GetHashCode()) * 397) ^ m32.GetHashCode()) * 397) ^ m33.GetHashCode();
            }
        }

        public bool Equals(mat4 rhs) => ((((m00.Equals(rhs.m00) && m01.Equals(rhs.m01)) && (m02.Equals(rhs.m02) && m03.Equals(rhs.m03))) && ((m10.Equals(rhs.m10) && m11.Equals(rhs.m11)) && (m12.Equals(rhs.m12) && m13.Equals(rhs.m13)))) 
            && (((m20.Equals(rhs.m20) && m21.Equals(rhs.m21)) && (m22.Equals(rhs.m22) && m23.Equals(rhs.m23))) && ((m30.Equals(rhs.m30) && m31.Equals(rhs.m31)) && (m32.Equals(rhs.m32) && m33.Equals(rhs.m33)))));

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is mat4 && Equals((mat4) obj);
        }

        public static bool operator ==(mat4 lhs, mat4 rhs) => lhs.Equals(rhs);
        public static bool operator !=(mat4 lhs, mat4 rhs) => !lhs.Equals(rhs);

        public mat4 Transposed => new mat4(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);

        // m00 m01 m02 m03
        // m10 m11 m12 m13
        // m20 m21 m22 m23
        // m30 m31 m32 m33
        public float Determinant()
        {     
            return   m00 * (m11 * (m22 * m33 - m32 * m23) - m21 * (m12 * m33 - m32 * m13) + m31 * (m12 * m23 - m22 * m13)) 
                   - m10 * (m01 * (m22 * m33 - m32 * m23) - m21 * (m02 * m33 - m32 * m03) + m31 * (m02 * m23 - m22 * m03)) 
                   + m20 * (m01 * (m12 * m33 - m32 * m13) - m11 * (m02 * m33 - m32 * m03) + m31 * (m02 * m13 - m12 * m03)) 
                   - m30 * (m01 * (m12 * m23 - m22 * m13) - m11 * (m02 * m23 - m22 * m03) + m21 * (m02 * m13 - m12 * m03));
        }

        public mat4 Inverse()
        {     
            // m11 m12 m13
            // m21 m22 m23
            // m31 m32 m33
            float C00 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);

            // m10 m12 m13
            // m20 m22 m23
            // m30 m32 m33
            float C01 = m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30);

            // m10 m11 m13
            // m20 m21 m23
            // m30 m31 m33
            float C02 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30);

            // m10 m11 m12 
            // m20 m21 m22 
            // m30 m31 m32 
            float C03 = m10 * (m21 * m33 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30);

            // m01 m02 m03 
            // m21 m22 m23
            // m31 m32 m33
            float C10 = m01 * (m22 * m33 - m22 * m32) - m02 * (m21 * m33 - m23 * m31) + m03 * (m21 * m32 - m22 * m31);

            // m00 m02 m03
            // m20 m22 m23
            // m30 m32 m33
            float C11 = m00 * (m22 * m33 - m23 * m32) - m02 * (m20 * m33 - m23 * m30) + m03 * (m20 * m32 - m22 * m30);

            // m00 m01 m03
            // m20 m21 m23
            // m30 m31 m33
            float C12 = m00 * (m21 * m33 - m23 * m31) - m01 * (m20 * m33 - m23 * m30) + m03 * (m20 * m31 - m21 * m30);

            // m00 m01 m02
            // m20 m21 m22
            // m30 m31 m32
            float C13 = m00 * (m21 * m32 - m22 * m31) - m01 * (m20 * m32 - m22 * m30) + m02 * (m20 * m31 - m21 * m30);

            // m01 m02 m03
            // m11 m12 m13
            // m31 m32 m33
            float C20 = m01 * (m12 * m33 - m13 * m32) - m02 * (m11 * m33 - m13 * m31) + m03 * (m11 * m32 - m12 * m31);

            // m00 m02 m03
            // m10 m12 m13
            // m30 m32 m33
            float C21 = m00 * (m12 * m33 - m13 * m32) - m02 * (m10 * m33 - m13 * m30) + m03 * (m10 * m32 - m12 * m30);

            // m00 m01 m03
            // m10 m11 m13
            // m30 m31 m33
            float C22 = m00 * (m10 * m33 - m13 * m30) - m01 * (m10 * m33 - m13 * m30) + m03 * (m10 * m31 - m11 * m30);

            // m00 m01 m02
            // m10 m11 m12
            // m30 m31 m32
            float C23 = m00 * (m11 * m32 - m12 * m31) - m01 * (m10 * m32 - m12 * m30) + m02 * (m10 * m31 - m11 * m30);

            // m01 m02 m03
            // m11 m12 m13
            // m21 m22 m23
            float C30 = m01 * (m12 * m23 - m13 * m22) - m02 * (m11 * m23 - m13 * m21) + m03 * (m11 * m22 - m12 * m21);

            // m00 m02 m03
            // m10 m12 m13
            // m20 m22 m23
            float C31 = m00 * (m12 * m23 - m13 * m22) - m02 * (m10 * m23 - m13 * m20) + m03 * (m10 * m22 - m12 * m20);

            // m00 m01 m03
            // m10 m11 m13
            // m20 m21 m23
            float C32 = m00 * (m11 * m23 - m13 * m21) - m01 * (m10 * m23 - m13 * m20) + m03 * (m10 * m21 - m11 * m20);

            // m00 m01 m02
            // m10 m11 m12
            // m20 m21 m22
            float C33 = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);

            float lDet = (m00 * C00) - (m01 * C01) + (m02 * C02) - (m03 * C03);

            return new mat4(
                C00 / lDet, C10 / lDet, C20 / lDet, C30 / lDet,
                C01 / lDet, C11 / lDet, C21 / lDet, C31 / lDet,
                C02 / lDet, C12 / lDet, C22 / lDet, C32 / lDet,
                C03 / lDet, C13 / lDet, C23 / lDet, C33 / lDet
            );
        }

        public static mat4 operator*(mat4 lhs, float rhs) => 
            new mat4(
                rhs * lhs.m00, rhs * lhs.m10, rhs * lhs.m20, rhs * lhs.m30, 
                rhs * lhs.m01, rhs * lhs.m11, rhs * lhs.m21, rhs * lhs.m31, 
                rhs * lhs.m02, rhs * lhs.m12, rhs * lhs.m22, rhs * lhs.m32, 
                rhs * lhs.m03, rhs * lhs.m13, rhs * lhs.m23, rhs * lhs.m33
            );

        public static mat4 operator*(float lhs, mat4 rhs) => 
            new mat4(
                lhs * rhs.m00, lhs * rhs.m10, lhs * rhs.m20, lhs * rhs.m30, 
                lhs * rhs.m01, lhs * rhs.m11, lhs * rhs.m21, lhs * rhs.m31, 
                lhs * rhs.m02, lhs * rhs.m12, lhs * rhs.m22, lhs * rhs.m32, 
                lhs * rhs.m03, lhs * rhs.m13, lhs * rhs.m23, lhs * rhs.m33
            );

        public static mat4 operator/(mat4 lhs, float rhs) => 
            new mat4(
                rhs / lhs.m00, rhs / lhs.m10, rhs / lhs.m20, rhs / lhs.m30, 
                rhs / lhs.m01, rhs / lhs.m11, rhs / lhs.m21, rhs / lhs.m31, 
                rhs / lhs.m02, rhs / lhs.m12, rhs / lhs.m22, rhs / lhs.m32, 
                rhs / lhs.m03, rhs / lhs.m13, rhs / lhs.m23, rhs / lhs.m33
            );

        public static mat4 operator/(float lhs, mat4 rhs) => 
            new mat4(
                lhs / rhs.m00, lhs / rhs.m10, lhs / rhs.m20, lhs / rhs.m30, 
                lhs / rhs.m01, lhs / rhs.m11, lhs / rhs.m21, lhs / rhs.m31, 
                lhs / rhs.m02, lhs / rhs.m12, lhs / rhs.m22, lhs / rhs.m32, 
                lhs / rhs.m03, lhs / rhs.m13, lhs / rhs.m23, lhs / rhs.m33
            );

        public static mat4 operator+(mat4 lhs, mat4 rhs) => 
            new mat4(
                lhs.m00 + rhs.m00, lhs.m10 + rhs.m10, lhs.m20 + rhs.m20, lhs.m30 + rhs.m30, 
                lhs.m01 + rhs.m01, lhs.m11 + rhs.m11, lhs.m21 + rhs.m21, lhs.m31 + rhs.m31, 
                lhs.m02 + rhs.m02, lhs.m12 + rhs.m12, lhs.m22 + rhs.m22, lhs.m32 + rhs.m32, 
                lhs.m03 + rhs.m03, lhs.m13 + rhs.m13, lhs.m23 + rhs.m23, lhs.m33 + rhs.m33);

        public static mat4 operator-(mat4 lhs, mat4 rhs) => 
            new mat4(
                lhs.m00 - rhs.m00, lhs.m10 - rhs.m10, lhs.m20 - rhs.m20, lhs.m30 - rhs.m30, 
                lhs.m01 - rhs.m01, lhs.m11 - rhs.m11, lhs.m21 - rhs.m21, lhs.m31 - rhs.m31, 
                lhs.m02 - rhs.m02, lhs.m12 - rhs.m12, lhs.m22 - rhs.m22, lhs.m32 - rhs.m32, 
                lhs.m03 - rhs.m03, lhs.m13 - rhs.m13, lhs.m23 - rhs.m23, lhs.m33 - rhs.m33);

        public static mat4 operator*(mat4 lhs, mat4 rhs) => 
            new mat4(
                lhs.m00 * rhs.m00 + lhs.m01 * rhs.m10 + lhs.m02 * rhs.m20 + lhs.m03 * rhs.m30, 
                lhs.m10 * rhs.m00 + lhs.m11 * rhs.m10 + lhs.m12 * rhs.m20 + lhs.m13 * rhs.m30, 
                lhs.m20 * rhs.m00 + lhs.m21 * rhs.m10 + lhs.m22 * rhs.m20 + lhs.m23 * rhs.m30, 
                lhs.m30 * rhs.m00 + lhs.m31 * rhs.m10 + lhs.m32 * rhs.m20 + lhs.m33 * rhs.m30, 

                lhs.m00 * rhs.m01 + lhs.m01 * rhs.m11 + lhs.m02 * rhs.m21 + lhs.m03 * rhs.m31, 
                lhs.m10 * rhs.m01 + lhs.m11 * rhs.m11 + lhs.m12 * rhs.m21 + lhs.m13 * rhs.m31, 
                lhs.m20 * rhs.m01 + lhs.m21 * rhs.m11 + lhs.m22 * rhs.m21 + lhs.m23 * rhs.m31, 
                lhs.m30 * rhs.m01 + lhs.m31 * rhs.m11 + lhs.m32 * rhs.m21 + lhs.m33 * rhs.m31, 

                lhs.m00 * rhs.m02 + lhs.m01 * rhs.m12 + lhs.m02 * rhs.m22 + lhs.m03 * rhs.m32, 
                lhs.m10 * rhs.m02 + lhs.m11 * rhs.m12 + lhs.m12 * rhs.m22 + lhs.m13 * rhs.m32, 
                lhs.m20 * rhs.m02 + lhs.m21 * rhs.m12 + lhs.m22 * rhs.m22 + lhs.m23 * rhs.m32, 
                lhs.m30 * rhs.m02 + lhs.m31 * rhs.m12 + lhs.m32 * rhs.m22 + lhs.m33 * rhs.m32, 

                lhs.m00 * rhs.m03 + lhs.m01 * rhs.m13 + lhs.m02 * rhs.m23 + lhs.m03 * rhs.m33, 
                lhs.m10 * rhs.m03 + lhs.m11 * rhs.m13 + lhs.m12 * rhs.m23 + lhs.m13 * rhs.m33, 
                lhs.m20 * rhs.m03 + lhs.m21 * rhs.m13 + lhs.m22 * rhs.m23 + lhs.m23 * rhs.m33, 
                lhs.m30 * rhs.m03 + lhs.m31 * rhs.m13 + lhs.m32 * rhs.m23 + lhs.m33 * rhs.m33 );


        public static vec4 operator*(mat4 lhs, vec4 rhs) => 
            new vec4(
                lhs.m00 * rhs.x + lhs.m01 * rhs.y + lhs.m02 * rhs.z + lhs.m03 * rhs.w, 
                lhs.m10 * rhs.x + lhs.m11 * rhs.y + lhs.m12 * rhs.z + lhs.m13 * rhs.w, 
                lhs.m20 * rhs.x + lhs.m21 * rhs.y + lhs.m22 * rhs.z + lhs.m23 * rhs.w, 
                lhs.m30 * rhs.x + lhs.m31 * rhs.y + lhs.m32 * rhs.z + lhs.m33 * rhs.w );


        public static mat4 Rotation(float angle, vec3 v)
        {
            var c = (float)System.Math.Cos((double)angle);
            var s = (float)System.Math.Sin((double)angle);
        
            var axis = v.Normalized;
            var temp = (1 - c) * axis;
        
            var m = new mat4(1.0f);
            m.m00 = c + temp.x * axis.x;
            m.m10 = 0 + temp.x * axis.y + s * axis.z;
            m.m20 = 0 + temp.x * axis.z - s * axis.y;
        
            m.m01 = 0 + temp.y * axis.x - s * axis.z;
            m.m11 = c + temp.y * axis.y;
            m.m21 = 0 + temp.y * axis.z + s * axis.x;
        
            m.m02 = 0 + temp.z * axis.x + s * axis.y;
            m.m12 = 0 + temp.z * axis.y - s * axis.x;
            m.m22 = c + temp.z * axis.z;
            return m;
        }

        public static mat4 Scaling(float x, float y, float z)
        {
            var m = new mat4(1.0f);
            m.m00 = x;
            m.m11 = y;
            m.m22 = z;
            return m;
        }
        public static mat4 Scaling(vec3 v) => Scaling(v.x, v.y, v.z);
        public static mat4 Scaling(float s) => Scaling(s, s, s);

        public static mat4 Translation(float x, float y, float z)
        {
            var m = new mat4(1.0f);
            m.m03 = x;
            m.m13 = y;
            m.m23 = z;
            return m;
        }
        
        public static mat4 Translation(vec3 v) => Translation(v.x, v.y, v.z);

        public static mat4 LookAt(vec3 aEye, vec3 aCenter, vec3 aUp)
        {
            var f = (aCenter - aEye).Normalized;
            var s = f.Cross(aUp).Normalized;
            var u = s.Cross(f);
            var m = new mat4(1.0f);
            m.m00 = s.x;
            m.m01 = s.y;
            m.m02 = s.z;
            m.m10 = u.x;
            m.m11 = u.y;
            m.m12 = u.z;
            m.m20 = -f.x;
            m.m21 = -f.y;
            m.m22 = -f.z;
            m.m03 = -s.Dot(aEye);
            m.m13 = -u.Dot(aEye);
            m.m23 = f.Dot(aEye);

            return m;
        }

        public static mat4 Ortho(float aLeft, float aRight, float aBottom, float aTop, float aNear, float aFar)
        {
            var m = new mat4(1.0f);
            m.m00 =  2 / (aRight - aLeft);
            m.m11 =  2 / (aTop - aBottom);
            m.m22 = -2 / (aFar - aNear);
            m.m03 = -(aRight + aLeft) / (aRight - aLeft);
            m.m13 = -(aTop + aBottom) / (aTop - aBottom);
            m.m23 = -(aFar + aNear) / (aFar - aNear);

            return m;
        }

        public static mat4 Ortho(float aLeft, float aRight, float aBottom, float aTop)
        {
            var m = new mat4(1.0f);
            m.m00 =  2 / (aRight - aLeft);
            m.m11 =  2 / (aTop - aBottom);
            m.m22 = -1;
            m.m03 = -(aRight + aLeft)/(aRight - aLeft);
            m.m13 = -(aTop + aBottom)/(aTop - aBottom);

            return m;
        }
    }
}