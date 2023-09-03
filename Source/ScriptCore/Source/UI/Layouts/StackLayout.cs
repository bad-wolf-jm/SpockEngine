using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIStackLayout : UIComponent
    {
        public UIStackLayout() : base(Interop.UIStackLayout_Create()) { }

        ~UIStackLayout() { Interop.UIStackLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, string aKey)
        {
            Interop.UIStackLayout_Add(mInstance, aChild.Instance, aKey);
        }

        public void SetCurrent(string aKey)
        {
            Interop.UIStackLayout_SetCurrent(mInstance, aKey);
        }
    }
}