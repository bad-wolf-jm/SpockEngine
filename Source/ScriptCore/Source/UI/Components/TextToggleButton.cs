using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UITextToggleButton : UILabel
    {
        public UITextToggleButton() : base(UITextToggleButton_Create(), true) { }

        public UITextToggleButton(string aText) : base(UITextToggleButton_CreateWithText(aText), true) { }

        ~UITextToggleButton() { UITextToggleButton_Destroy(mInstance); }

        public bool Active
        {
            get { return UITextToggleButton_IsActive(mInstance); }
            set { UITextToggleButton_SetActive(mInstance, value); }
        }

        public void SetActiveColor(Math.vec4 aColor)
        {
            UITextToggleButton_SetActiveColor(mInstance, aColor);
        }

        public void SetInactiveColor(Math.vec4 aColor)
        {
            UITextToggleButton_SetInactiveColor(mInstance, aColor);
        }

        public delegate bool OnClickDelegate(bool aValue);
        OnClickDelegate onClicked;
        public void OnClicked(OnClickDelegate aHandler)
        {
            onClicked = aHandler;

            UITextToggleButton_OnClicked(mInstance, Marshal.GetFunctionPointerForDelegate(onClicked));
        }

        public delegate bool OnChangeDelegate();
        OnChangeDelegate onChanged;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;
            
            UITextToggleButton_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextToggleButton_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_OnClicked(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_OnChanged(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UITextToggleButton_IsActive(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetActive(ulong aInstance, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetActiveColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextToggleButton_SetInactiveColor(ulong aInstance, Math.vec4 aColor);
    }
}
