using System;

namespace SpockEngine
{
    public class Component
    {
        public Component() {}
    }

    public class sNodeTransformComponent : Component
    {
        public mat4 mMatrix;

        public sNodeTransformComponent() 
        { 
            Console.WriteLine($"New sNodeTransformComponent instantiated!!");
        }

        public sNodeTransformComponent( ref mat4 aMatrix ) 
        { 
            mMatrix = aMatrix;

            Console.WriteLine($"New sNodeTransformComponent instantiated with matrix!!");
            Console.WriteLine($"{mMatrix.m00} {mMatrix.m01} {mMatrix.m02} {mMatrix.m03}");
            Console.WriteLine($"{mMatrix.m10} {mMatrix.m11} {mMatrix.m12} {mMatrix.m13}");
            Console.WriteLine($"{mMatrix.m20} {mMatrix.m21} {mMatrix.m22} {mMatrix.m23}");
            Console.WriteLine($"{mMatrix.m30} {mMatrix.m31} {mMatrix.m32} {mMatrix.m33}");

        }

    }

}
