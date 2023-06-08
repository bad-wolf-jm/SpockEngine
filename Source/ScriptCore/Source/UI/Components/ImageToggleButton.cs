using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIImageToggleButton : UIComponent
    {
        public UIImageToggleButton() : base(Interop.UIImageToggleButton_Create()) { }

        ~UIImageToggleButton() { Interop.UIImageToggleButton_Destroy(mInstance); }

        public bool Active 
        {
            get { return Interop.UIImageToggleButton_IsActive(mInstance); }
            set { Interop.UIImageToggleButton_SetActive(mInstance, value); }
        }

        public void SetActiveImage(UIBaseImage aImage)
        {
            Interop.UIImageToggleButton_SetActiveImage(mInstance, aImage.Instance);
        }

        public void SetInactiveImage(UIBaseImage aImage)
        {
            Interop.UIImageToggleButton_SetInactiveImage(mInstance, aImage.Instance);
        }

        public delegate bool OnClickDelegate(bool aValue);
        OnClickDelegate onClicked;
        public void OnClicked(OnClickDelegate aHandler)
        {
            onClicked = aHandler;
            
            Interop.UIImageToggleButton_OnClicked(mInstance, Marshal.GetFunctionPointerForDelegate(onClicked));
        }

        public delegate bool OnChangeDelegate();
        OnChangeDelegate onChanged;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;
            
            Interop.UIImageToggleButton_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }
    }
}
