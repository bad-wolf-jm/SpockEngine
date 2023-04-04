using System;
using System.Runtime.CompilerServices;

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

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIComboBox_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIComboBox_CreateWithItems(string[] aItems);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static int UIComboBox_GetCurrent(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_SetCurrent(ulong aInstance, int aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIComboBox_SetItemList(ulong aInstance, string[] aItems);
    }
}
