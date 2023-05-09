using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDropdownButton : UIComponent
    {
        public UIDropdownButton() : base(UIDropdownButton_Create()) { }

        ~UIDropdownButton() { UIDropdownButton_Destroy(mInstance); }

        public string Text
        {
            set { UIDropdownButton_SetText(mInstance, value); }
        }

        public UIBaseImage Image
        {
            set { UIDropdownButton_SetImage(mInstance, value); }
        }

        public Math.vec4 TextColor
        {
            set { UIDropdownButton_SetTextColor(mInstance, value); }
        }

        public UIComponent Content
        {
            set { UIDropdownButton_SetContent(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDropdownButton_Create();


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIDropdownButton_SetContent(ulong aInstance, UIComponent aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetImage(ulong aInstance, UIBaseImage aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDropdownButton_SetTextColor(ulong aInstance, Math.vec4 aColor);
    }
}
