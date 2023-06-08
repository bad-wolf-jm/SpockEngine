using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UITextToggleButton : UILabel
    {
        public UITextToggleButton() : base(Interop.UITextToggleButton_Create(), true) { }

        public UITextToggleButton(string aText) : base(Interop.UITextToggleButton_CreateWithText(aText), true) { }

        ~UITextToggleButton() { Interop.UITextToggleButton_Destroy(mInstance); }

        public bool Active
        {
            get { return Interop.UITextToggleButton_IsActive(mInstance); }
            set { Interop.UITextToggleButton_SetActive(mInstance, value); }
        }

        public void SetActiveColor(Math.vec4 aColor)
        {
            Interop.UITextToggleButton_SetActiveColor(mInstance, aColor);
        }

        public void SetInactiveColor(Math.vec4 aColor)
        {
            Interop.UITextToggleButton_SetInactiveColor(mInstance, aColor);
        }

        public delegate bool OnClickDelegate(bool aValue);
        OnClickDelegate onClicked;
        public void OnClicked(OnClickDelegate aHandler)
        {
            onClicked = aHandler;

            Interop.UITextToggleButton_OnClicked(mInstance, Marshal.GetFunctionPointerForDelegate(onClicked));
        }

        public delegate bool OnChangeDelegate();
        OnChangeDelegate onChanged;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;
            
            Interop.UITextToggleButton_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }
    }
}
