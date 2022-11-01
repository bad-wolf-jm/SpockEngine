using System;

namespace SpockEngine
{
    public class Component
    {
        public Component() {}
    }

    public class sNodeTransformComponent : Component
    {
        private int mDummy;

        public sNodeTransformComponent() 
        { 
            mDummy = 123; 
            Console.WriteLine("New sNodeTransformComponent instantiated!!");
        }
    }

}
