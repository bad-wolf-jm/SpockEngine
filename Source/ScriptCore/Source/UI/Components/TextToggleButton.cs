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
    }
}
