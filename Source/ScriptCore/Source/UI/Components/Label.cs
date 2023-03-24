using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UILabel : UIComponent
    {
        public UILabel() : base(UILabel_Create()) { }
        public UILabel(string aText) : base(UILabel_CreateWithText(aText)) { }

        ~UILabel() { UILabel_Destroy(mInstance); }

        public void SetText(string aText) { UILabel_SetText(mInstance, aText); }

        public void SetTextColor(Math.vec4 aColor) { UILabel_SetTextColor(mInstance, aColor); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UILabel_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UILabel_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UILabel_SetTextColor(ulong aInstance, Math.vec4 aText);
    }
}
