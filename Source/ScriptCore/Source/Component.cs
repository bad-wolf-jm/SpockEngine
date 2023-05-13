using System;
using SpockEngine.Math;

namespace SpockEngine
{
    public class Component
    {
        public Component() { }
    }

    public class sTag : Component
    {
        public string mValue;

        public sTag()
        {
        }

        public sTag(ref string aValue)
        {
            mValue = aValue;
        }
    }
}
