using System;
using System.Runtime.CompilerServices;

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

        public delegate bool OnChangeDelegate(bool aValue);
        OnChangeDelegate onChanged;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;
            
            UIImageToggleButton_OnChanged(mInstance, onChanged);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_OnChanged(ulong aInstance, OnChangeDelegate aHandler);

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
