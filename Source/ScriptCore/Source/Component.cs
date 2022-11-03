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

        public sNodeTransformComponent( mat4 aMatrix ) 
        { 
            mMatrix = aMatrix;
        }

    }

    public class sTag : Component
    {
        public string mValue;

        public sTag() 
        { 
            Console.WriteLine($"New sTag instantiated!!");
        }

        public sTag( string aValue ) 
        { 
            mValue = aValue;
        }

    }

}
