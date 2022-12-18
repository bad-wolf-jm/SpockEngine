using System;
using SpockEngine.Math;

namespace SpockEngine
{
    public class Component
    {
        public Component() { }
    }

    public class sNodeTransformComponent : Component
    {
        public mat4 mMatrix;

        public sNodeTransformComponent()
        {
        }

        public sNodeTransformComponent(mat4 aMatrix)
        {
            mMatrix = aMatrix;
        }
    }

    public class sTransformMatrixComponent : Component
    {
        public mat4 mMatrix;

        public sTransformMatrixComponent()
        {
        }

        public sTransformMatrixComponent(mat4 aMatrix)
        {
            mMatrix = aMatrix;
        }
    }

    public class sTag : Component
    {
        public string mValue;

        public sTag()
        {
        }

        public sTag(string aValue)
        {
            mValue = aValue;
        }
    }

    public enum eLightType
    {
        DIRECTIONAL = 0,
        SPOTLIGHT = 1,
        POINT_LIGHT = 2
    }

    public class sLightComponent : Component
    {
        public eLightType mType;

        public float mIntensity;

        public vec3 mColor;

        public float mCone;

        public sLightComponent()
        {
        }

        public sLightComponent(eLightType aType, float aIntensity, vec3 aColor, float aCone)
        {
            mType = aType;
            mIntensity = aIntensity;
            mColor = aColor;
            mCone = aCone;
        }
    }

}
