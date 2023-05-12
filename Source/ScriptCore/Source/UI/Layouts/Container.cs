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

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIContainer_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIContainer_Destroy(ulong aInstance);
        
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIContainer_SetContent(ulong aInstance, ulong aChild);
    }
}