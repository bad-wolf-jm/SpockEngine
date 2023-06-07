using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIImageToggleButton : UIComponent
    {
        public UIImageToggleButton() : base(UIImageToggleButton_Create()) { }

        ~UIImageToggleButton() { UIImageToggleButton_Destroy(mInstance); }

        public bool Active 
        {
            get { return UIImageToggleButton_IsActive(mInstance); }
            set { UIImageToggleButton_SetActive(mInstance, value); }
        }

        public void SetActiveImage(UIBaseImage aImage)
        {
            UIImageToggleButton_SetActiveImage(mInstance, aImage.Instance);
        }

        public void SetInactiveImage(UIBaseImage aImage)
        {
            UIImageToggleButton_SetInactiveImage(mInstance, aImage.Instance);
        }

        public delegate bool OnClickDelegate(bool aValue);
        OnClickDelegate onClicked;
        public void OnClicked(OnClickDelegate aHandler)
        {
            onClicked = aHandler;
            
            UIImageToggleButton_OnClicked(mInstance, Marshal.GetFunctionPointerForDelegate(onClicked));
        }

        public delegate bool OnChangeDelegate();
        OnChangeDelegate onChanged;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;
            
            UIImageToggleButton_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_OnClicked(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_OnChanged(ulong aInstance, IntPtr aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIImageToggleButton_IsActive(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetActive(ulong aInstance, bool aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetActiveImage(ulong aInstance, ulong aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetInactiveImage(ulong aInstance, ulong aImage);
    }
}
