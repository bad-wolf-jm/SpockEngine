using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIContainer : UIComponent
    {
        public UIContainer() : base(UIContainer_Create()) { }

        ~UIContainer() { UIContainer_Destroy(mInstance); }

        public void SetContent(UIComponent aChild)
        {
            UIContainer_SetContent(mInstance, aChild.Instance);
        }
    }
}