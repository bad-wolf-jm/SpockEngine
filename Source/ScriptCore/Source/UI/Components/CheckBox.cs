using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UICheckBox : UIComponent
    {
        public UICheckBox() : base(UICheckBox_Create()) { }

        ~UICheckBox() { UICheckBox_Destroy(mInstance); }

        public void OnClick(Math.vec4 aColor) { UICheckBox_OnClick(mInstance, aColor); }

        public bool IsChecked
        {
            get { return UICheckBox_IsChecked(mInstance); }
            set { UICheckBox_SetIsChecked(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UICheckBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_OnClick(ulong aInstance, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_IsChecked(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UICheckBox_SetIsChecked(ulong aInstance, bool aValue);
    }
}
