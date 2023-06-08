using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIContainer : UIComponent
    {
        public UIContainer() : base(Interop.UIContainer_Create()) { }

        ~UIContainer() { Interop.UIContainer_Destroy(mInstance); }

        public void SetContent(UIComponent aChild)
        {
            Interop.UIContainer_SetContent(mInstance, aChild.Instance);
        }
    }
}