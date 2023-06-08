using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UICheckBox : UIComponent
    {
        public UICheckBox() : base(UICheckBox_Create()) { }

        ~UICheckBox() { UICheckBox_Destroy(mInstance); }


        public delegate void OnClickDelegate();
        OnClickDelegate onChanged;
        public void OnClick(OnClickDelegate aHandler)
        {
            onChanged = aHandler;

            UICheckBox_OnClick(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }

        public bool IsChecked
        {
            get { return UICheckBox_IsChecked(mInstance); }
            set { UICheckBox_SetIsChecked(mInstance, value); }
        }
    }
}
