using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIFileTree : UITreeView
    {
        public UIFileTree() : base(Interop.UIFileTree_Create(), true) { }

        ~UIFileTree() { Interop.UIFileTree_Destroy(mInstance); }

        public void Add(string aText) { Interop.UIFileTree_Add(mInstance, aText); }
        public void Remove(string aText) { Interop.UIFileTree_Remove(mInstance, aText); }

        public delegate void OnSelectedDelegate([MarshalAs(UnmanagedType.LPWStr)] string aPath);
        OnSelectedDelegate onSelected;
        public void OnSelected(OnSelectedDelegate aHandler)
        {
            onSelected = aHandler;

            Interop.UIFileTree_OnSelected(mInstance, Marshal.GetFunctionPointerForDelegate(onSelected));
        }


    }
}
