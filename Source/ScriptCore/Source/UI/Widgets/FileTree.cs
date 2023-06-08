using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIFileTree : UITreeView
    {
        public UIFileTree() : base(Interop.UIFileTree_Create(), true) { }

        ~UIFileTree() { Interop.UIFileTree_Destroy(mInstance); }

        public void Add(string aText) { Interop.UIFileTree_Add(mInstance, aText); }
    }
}
