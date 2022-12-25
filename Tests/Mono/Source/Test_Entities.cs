using System;
using SpockEngine;
using SpockEngine.Math;

namespace SEUnitTest
{
    public class EntityTest
    {
        public static bool TestHasTag(ref Entity aEntity)
        {
            return aEntity.Has<sTag>();
        }

        public static bool AddTagValue(ref Entity aEntity, ref string aTagValue)
        {
            aEntity.Add<sTag>(new sTag(ref aTagValue));

            return true;
        }

        public static bool TestTagValue(ref Entity aEntity, ref string aTagValue)
        {
            return aEntity.Get<sTag>().mValue.Equals(aTagValue);
        }

        public static bool TestHasNodeTransform(ref Entity aEntity)
        {
            return aEntity.Has<sNodeTransformComponent>();
        }

        public static bool TestNodeTransformValue(ref Entity aEntity, mat4 aMatrixValue)
        {
            return aEntity.Get<sNodeTransformComponent>().mMatrix == aMatrixValue;
        }

        public static bool AddNodeTransform(ref Entity aEntity, mat4 aMatrixValue)
        {
            aEntity.Add<sNodeTransformComponent>(new sNodeTransformComponent(aMatrixValue));

            return true;
        }

        public static bool TestHasTransformMatrix(ref Entity aEntity)
        {
            return aEntity.Has<sTransformMatrixComponent>();
        }

        public static bool TestTransformMatrixValue(ref Entity aEntity, mat4 aMatrixValue)
        {
            return aEntity.Get<sTransformMatrixComponent>().mMatrix == aMatrixValue;
        }

        public static bool AddNodeTransformMartix(ref Entity aEntity, mat4 aMatrixValue)
        {
            aEntity.Add<sTransformMatrixComponent>(new sTransformMatrixComponent(aMatrixValue));

            return true;
        }

        public static bool TestHasLight(ref Entity aEntity)
        {
            return aEntity.Has<sLightComponent>();
        }
        public static bool AddLight(ref Entity aEntity)
        {
            aEntity.Add<sLightComponent>(new sLightComponent());

            return true;
        }

    }
}
