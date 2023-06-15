using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIComboBox : UIComponent
    {
        public UIComboBox() : base(Interop.UIComboBox_Create()) { }
        public UIComboBox(string[] aItems) : this()
        {
            SetItemList(aItems);
        }

        ~UIComboBox() { Interop.UIComboBox_Destroy(mInstance); }

        public int CurrentItem
        {
            get { return Interop.UIComboBox_GetCurrent(mInstance); }
            set { Interop.UIComboBox_SetCurrent(mInstance, value); }
        }

        public void SetItemList(string[] aItems)
        {
            Interop.UIComboBox_SetItemList(mInstance, aItems, aItems.Length);
        }

        public delegate void ChangedDelegate(int aIndex);
        ChangedDelegate onChanged;
        public void OnChanged(ChangedDelegate aHandler)
        {
            onChanged = aHandler;

            Interop.UIComboBox_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }
    }
}
