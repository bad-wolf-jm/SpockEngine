using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIWorkspaceDocument : UIComponent
    {

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspaceDocument_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_RegisterSaveDelegate(ulong aInstance, DocumentSaveDelegate aDelegate);
        public delegate bool DocumentSaveDelegate();

        public UIWorkspaceDocument() : base(UIWorkspaceDocument_Create())
        {
            UIWorkspaceDocument_RegisterSaveDelegate(mInstance, DoSave);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Destroy(ulong aInstance);
        ~UIWorkspaceDocument() { UIWorkspaceDocument_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_SetContent(ulong aInstance, ulong aContent);
        public void SetContent(UIComponent aContent) { UIWorkspaceDocument_SetContent(mInstance, aContent.Instance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspaceDocument_SetName(ulong aInstance, string aName);
        public void SetName(string aName) { UIWorkspaceDocument_SetName(mInstance, aName); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Update(ulong aInstance);
        public void Update() { UIWorkspaceDocument_Update(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static bool UIWorkspaceDocument_IsDirty(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_MarkAsDirty(ulong aInstance, bool aDirty);
        public bool IsDirty
        {
            get { return UIWorkspaceDocument_IsDirty(mInstance); }
            set { UIWorkspaceDocument_MarkAsDirty(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_Open(ulong aInstance);
        public void Open() { UIWorkspaceDocument_IsDirty(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_RequestClose(ulong aInstance);
        public void RequestClose() { UIWorkspaceDocument_RequestClose(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspaceDocument_ForceClose(ulong aInstance);
        public void ForceClose() { UIWorkspaceDocument_ForceClose(mInstance); }

        public virtual bool DoSave() { return true; }
    }

    public class UIWorkspace : UIComponent
    {
        private List<UIWorkspaceDocument> mDocuments;

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIWorkspace_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_RegisterCloseDocumentDelegate(ulong aInstance, DocumentCloseDelegate aDelegate);
        public delegate void DocumentCloseDelegate(ulong[] aDocumentList);

        public UIWorkspace() : base(UIWorkspace_Create()) { 
            mDocuments = new List<UIWorkspaceDocument>(); 

            UIWorkspace_RegisterCloseDocumentDelegate(mInstance, CloseDocuments);
        }

        private void CloseDocuments(ulong[] aPtrList) 
        { 
            mDocuments = mDocuments.FindAll(d => aPtrList.Contains(d.Instance));
        }

        public UIWorkspace(ulong aSelf) : base(aSelf) { mDocuments = new List<UIWorkspaceDocument>(); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Destroy(ulong aInstance);
        ~UIWorkspace() { UIWorkspace_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIWorkspace_Add(ulong aInstance, ulong aDocument);
        public void Add(UIWorkspaceDocument aDocument)
        {
            mDocuments.Add(aDocument);

            UIWorkspace_Add(mInstance, aDocument.Instance);
        }
    }
}
