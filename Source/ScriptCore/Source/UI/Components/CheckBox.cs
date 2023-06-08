using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UICheckBox : UIComponent
    {
        public UICheckBox() : base(Interop.UICheckBox_Create()) { }

        ~UICheckBox() { Interop.UICheckBox_Destroy(mInstance); }


        public delegate void OnClickDelegate();
        OnClickDelegate onChanged;
        public void OnClick(OnClickDelegate aHandler)
        {
            onChanged = aHandler;

            Interop.UICheckBox_OnClick(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }

        public bool IsChecked
        {
            get { return Interop.UICheckBox_IsChecked(mInstance); }
            set { Interop.UICheckBox_SetIsChecked(mInstance, value); }
        }
    }
}
