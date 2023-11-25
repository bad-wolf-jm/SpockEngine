using System.Runtime.InteropServices;

namespace SimulationEngine.Editor.Core;

[StructLayout(LayoutKind.Sequential)]
public struct mat3
{
    public float m00;
    public float m10;
    public float m20;

    public float m01;
    public float m11;
    public float m21;

    public float m02;
    public float m12;
    public float m22;

    public mat3(float aDiag)
    {
        m00 = aDiag; m01 = 0.0f; m02 = 0.0f;
        m10 = 0.0f; m11 = aDiag; m12 = 0.0f;
        m20 = 0.0f; m21 = 0.0f; m22 = aDiag;
    }

    public mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
    {
        m00 = a00; m01 = a01; m02 = a02;
        m10 = a10; m11 = a11; m12 = a12;
        m20 = a20; m21 = a21; m22 = a22;
    }

    public mat3(mat4 aOther)
    {
        m00 = aOther.m00; m01 = aOther.m01; m02 = aOther.m02;
        m10 = aOther.m10; m11 = aOther.m11; m12 = aOther.m12;
        m20 = aOther.m20; m21 = aOther.m21; m22 = aOther.m22;
    }

    public float[,] Values => new[,] { { m00, m10, m20 }, { m01, m11, m21 }, { m02, m12, m22 } };

    public vec3 mC0
    {
        get { return new vec3(m00, m10, m20); }
        set { m00 = value.x; m10 = value.y; m20 = value.z; }
    }

    public vec3 mC1
    {
        get { return new vec3(m01, m11, m21); }
        set { m01 = value.x; m11 = value.y; m21 = value.z; }
    }

    public vec3 mC2
    {
        get { return new vec3(m02, m12, m22); }
        set { m02 = value.x; m12 = value.y; m22 = value.z; }
    }

    public bool Equals(mat3 rhs) => ((((m00.Equals(rhs.m00) && m01.Equals(rhs.m01)) && m02.Equals(rhs.m02)) && (m10.Equals(rhs.m10) && m11.Equals(rhs.m11))) && ((m12.Equals(rhs.m12) && m20.Equals(rhs.m20)) && (m21.Equals(rhs.m21) && m22.Equals(rhs.m22))));

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        return obj is mat3 && Equals((mat3)obj);
    }

    public static bool operator ==(mat3 lhs, mat3 rhs) => lhs.Equals(rhs);
    public static bool operator !=(mat3 lhs, mat3 rhs) => !lhs.Equals(rhs);

    public override int GetHashCode()
    {
        unchecked
        {
            return ((((((((((((((((m00.GetHashCode()) * 397) ^ m01.GetHashCode()) * 397) ^ m02.GetHashCode()) * 397) ^ m10.GetHashCode()) * 397) ^ m11.GetHashCode()) * 397) ^ m12.GetHashCode()) * 397) ^ m20.GetHashCode()) * 397) ^ m21.GetHashCode()) * 397) ^ m22.GetHashCode();
        }
    }

    public mat3 Transposed => new mat3(m00, m10, m20, m01, m11, m21, m02, m12, m22);

    // m00 m01 m02
    // m10 m11 m12
    // m20 m21 m22
    public float Determinant()
    {
        return m00 * (m11 * m22 - m12 * m21)
               - m10 * (m01 * m22 - m02 * m21)
               + m20 * (m01 * m12 - m02 * m11);
    }


    public mat3 Inverse()
    {
        // m11 m12
        // m21 m22
        float C00 = m11 * m22 - m12 * m21;

        // m10 m12
        // m20 m22
        float C01 = m10 * m22 - m12 * m20;

        // m10 m11
        // m20 m21
        float C02 = m10 * m21 - m11 * m20;

        // m01 m02
        // m21 m22
        float C10 = m01 * m22 - m02 * m21;

        // m00 m02
        // m20 m22
        float C11 = m00 * m22 - m02 * m20;

        // m00 m01
        // m20 m21
        float C12 = m00 * m21 - m01 * m20;

        // m01 m02
        // m11 m12
        float C20 = m01 * m12 - m02 * m11;

        // m00 m02
        // m10 m12
        float C21 = m00 * m12 - m02 * m10;

        // m00 m01
        // m10 m11
        float C22 = m00 * m11 - m01 * m10;

        float lDet = (m00 * C00) - (m01 * C01) + (m02 * C02);

        return new mat3(
            C00 / lDet,
           -C10 / lDet,
            C20 / lDet,

           -C01 / lDet,
            C11 / lDet,
           -C21 / lDet,
            
            C02 / lDet,
           -C12 / lDet,
            C22 / lDet
        );
    }

    public static mat3 operator *(mat3 lhs, float rhs) =>
        new mat3(
            rhs * lhs.m00,
            rhs * lhs.m01,
            rhs * lhs.m02,
            rhs * lhs.m10,
            rhs * lhs.m11,
            rhs * lhs.m12,
            rhs * lhs.m20,
            rhs * lhs.m21,
            rhs * lhs.m22
        );

    public static mat3 operator *(float lhs, mat3 rhs) =>
        new mat3(
            lhs * rhs.m00,
            lhs * rhs.m01,
            lhs * rhs.m02,
            lhs * rhs.m10,
            lhs * rhs.m11,
            lhs * rhs.m12,
            lhs * rhs.m20,
            lhs * rhs.m21,
            lhs * rhs.m22
        );

    public static mat3 operator /(mat3 lhs, float rhs) =>
        new mat3(
             lhs.m00 / rhs,
             lhs.m01 / rhs,
             lhs.m02 / rhs,
             lhs.m10 / rhs,
             lhs.m11 / rhs,
             lhs.m12 / rhs,
             lhs.m20 / rhs,
             lhs.m21 / rhs,
             lhs.m22 / rhs
        );

    public static mat3 operator /(float lhs, mat3 rhs) =>
        new mat3(
            lhs / rhs.m00,
            lhs / rhs.m01,
            lhs / rhs.m02,
            lhs / rhs.m10,
            lhs / rhs.m11,
            lhs / rhs.m12,
            lhs / rhs.m20,
            lhs / rhs.m21,
            lhs / rhs.m22
        );

    public static mat3 operator +(mat3 lhs, mat3 rhs) =>
        new mat3(
            lhs.m00 + rhs.m00,
            lhs.m01 + rhs.m01,
            lhs.m02 + rhs.m02,
            lhs.m10 + rhs.m10,
            lhs.m11 + rhs.m11,
            lhs.m12 + rhs.m12,
            lhs.m20 + rhs.m20,
            lhs.m21 + rhs.m21,
            lhs.m22 + rhs.m22
        );

    public static mat3 operator -(mat3 lhs, mat3 rhs) =>
        new mat3(
            lhs.m00 - rhs.m00,
            lhs.m01 - rhs.m01,
            lhs.m02 - rhs.m02,
            lhs.m10 - rhs.m10,
            lhs.m11 - rhs.m11,
            lhs.m12 - rhs.m12,
            lhs.m20 - rhs.m20,
            lhs.m21 - rhs.m21,
            lhs.m22 - rhs.m22
        );

    public static mat3 operator *(mat3 lhs, mat3 rhs) =>
        new mat3(
            lhs.m00 * rhs.m00 + lhs.m01 * rhs.m10 + lhs.m02 * rhs.m20,
            lhs.m00 * rhs.m01 + lhs.m01 * rhs.m11 + lhs.m02 * rhs.m21,
            lhs.m00 * rhs.m02 + lhs.m01 * rhs.m12 + lhs.m02 * rhs.m22,

            lhs.m10 * rhs.m00 + lhs.m11 * rhs.m10 + lhs.m12 * rhs.m20,
            lhs.m10 * rhs.m01 + lhs.m11 * rhs.m11 + lhs.m12 * rhs.m21,
            lhs.m10 * rhs.m02 + lhs.m11 * rhs.m12 + lhs.m12 * rhs.m22,

            lhs.m20 * rhs.m00 + lhs.m21 * rhs.m10 + lhs.m22 * rhs.m20,
            lhs.m20 * rhs.m01 + lhs.m21 * rhs.m11 + lhs.m22 * rhs.m21,
            lhs.m20 * rhs.m02 + lhs.m21 * rhs.m12 + lhs.m22 * rhs.m22
        );

    public static vec3 operator *(mat3 lhs, vec3 rhs) =>
        new vec3(
            lhs.m00 * rhs.x + lhs.m01 * rhs.y + lhs.m02 * rhs.z,
            lhs.m10 * rhs.x + lhs.m11 * rhs.y + lhs.m12 * rhs.z,
            lhs.m20 * rhs.x + lhs.m21 * rhs.y + lhs.m22 * rhs.z
        );
}