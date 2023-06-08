using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIComboBox : UIComponent
    {
        public UIComboBox() : base(UIComboBox_Create()) { }
        public UIComboBox(string[] aItems) : base(UIComboBox_CreateWithItems(aItems)) { }

        ~UIComboBox() { UIComboBox_Destroy(mInstance); }

        public int CurrentItem
        {
            get { return UIComboBox_GetCurrent(mInstance); }
            set { UIComboBox_SetCurrent(mInstance, value); }
        }

        public void SetItemList(string[] aItems)
        {
            UIComboBox_SetItemList(mInstance, aItems);
        }

        public delegate void ChangedDelegate(int aIndex);
        ChangedDelegate onChanged;
        public void OnChanged(ChangedDelegate aHandler)
        {
            onChanged = aHandler;

            UIComboBox_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChanged));
        }
    }
}
