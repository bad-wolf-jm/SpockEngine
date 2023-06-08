using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIFileTree : UITreeView
    {
        public UIFileTree() : base(UIFileTree_Create(), true) { }

        ~UIFileTree() { UIFileTree_Destroy(mInstance); }

        public void Add(string aText) { UIFileTree_Add(mInstance, aText); }
    }
}
