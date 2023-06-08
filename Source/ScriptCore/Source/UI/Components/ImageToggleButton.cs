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
    }
}
