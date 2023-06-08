using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIStackLayout : UIComponent
    {
        public UIStackLayout() : base(UIStackLayout_Create()) { }

        ~UIStackLayout() { UIStackLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, string aKey)
        {
            UIStackLayout_Add(mInstance, aChild.Instance, aKey);
        }

        public void SetCurrent(string aKey)
        {
            UIStackLayout_SetCurrent(mInstance, aKey);
        }
    }
}