using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIWorkspace : UIComponent
    {

        public UIWorkspace() : base(UIWorkspace_Create()) { }
        public UIWorkspace(ulong aSelf) : base(aSelf) { }

        ~UIWorkspace() { UIWorkspace_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspace_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Add(ulong aInstance, ulong aDocument);
    }
}
