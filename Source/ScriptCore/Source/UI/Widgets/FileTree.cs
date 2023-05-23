using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIFileTree : UITreeView
    {
        public UIFileTree() : base(UIFileTree_Create(), true) { }

        ~UIFileTree() { UIFileTree_Destroy(mInstance); }

        public void Add(string aText) { UIFileTree_Add(mInstance, aText); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIFileTree_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFileTree_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIFileTree_Add(ulong aInstance, string aText);
    }
}
