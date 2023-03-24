using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIButton : UIComponent
    {
        public UIButton() : base(UIButton_Create()) { }
        public UIButton(string aText) : base(UIButton_CreateWithText(aText)) { }

        ~UIButton() { UIButton_Destroy(mInstance); }

        public void SetText(string aText) { UIButton_SetText(mInstance, aText); }

        public void OnClick(Math.vec4 aColor) { UIButton_OnClick(mInstance, aColor); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIButton_OnClick(ulong aInstance, Math.vec4 aText);
    }
}
