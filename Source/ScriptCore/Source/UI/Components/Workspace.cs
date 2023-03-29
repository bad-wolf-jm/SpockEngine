using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIWorkspaceDocument : UIComponent
    {
        public UIWorkspaceDocument() : base(UIWorkspaceDocument_Create()) { }

        ~UIWorkspaceDocument() { UIWorkspaceDocument_Destroy(mInstance); }

        public void SetContent(UIComponent aContent) { UIWorkspaceDocument_SetContent(mInstance, aContent.Instance); }

        public void Update() { UIWorkspaceDocument_Update(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspaceDocument_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Update(ulong aInstance);
    }


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
