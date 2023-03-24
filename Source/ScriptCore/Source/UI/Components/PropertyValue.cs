using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIPropertyValue : UIComponent
    {
        public UIPropertyValue() : base(UIPropertyValue_Create()) { }
        public UIPropertyValue(string aText) : base(UIPropertyValue_CreateWithText(aText)) { }

        ~UIPropertyValue() { UIPropertyValue_Destroy(mInstance); }

        public void SetValue(string aText) { UIPropertyValue_SetValue(mInstance, aText); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPropertyValue_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIPropertyValue_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIPropertyValue_SetValue(ulong aInstance, string aText);
    }
}
