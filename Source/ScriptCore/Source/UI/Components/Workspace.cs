using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIWorkspaceDocument : UIComponent
    {
        public delegate bool DocumentSaveDelegate();

        public UIWorkspaceDocument() : base(Interop.UIWorkspaceDocument_Create())
        {
            Interop.UIWorkspaceDocument_RegisterSaveDelegate(mInstance, DoSave);
        }

        public bool IsDirty
        {
            get { return Interop.UIWorkspaceDocument_IsDirty(mInstance); }
            set { Interop.UIWorkspaceDocument_MarkAsDirty(mInstance, value); }
        }

        public void Open() { Interop.UIWorkspaceDocument_IsDirty(mInstance); }

        public void RequestClose() { Interop.UIWorkspaceDocument_RequestClose(mInstance); }

        public void ForceClose() { Interop.UIWorkspaceDocument_ForceClose(mInstance); }

        public virtual bool DoSave() { return true; }
    }

    public class UIWorkspace : UIComponent
    {
        private List<UIWorkspaceDocument> mDocuments;

        public delegate void DocumentCloseDelegate(ulong[] aDocumentList);

        public UIWorkspace() : base(Interop.UIWorkspace_Create()) { 
            mDocuments = new List<UIWorkspaceDocument>(); 

            Interop.UIWorkspace_RegisterCloseDocumentDelegate(mInstance, CloseDocuments);
        }

        private void CloseDocuments(ulong[] aPtrList) 
        { 
            mDocuments = mDocuments.FindAll(d => aPtrList.Contains(d.Instance));
        }

        public UIWorkspace(ulong aSelf) : base(aSelf) { mDocuments = new List<UIWorkspaceDocument>(); }

        ~UIWorkspace() { Interop.UIWorkspace_Destroy(mInstance); }

        public void Add(UIWorkspaceDocument aDocument)
        {
            mDocuments.Add(aDocument);

            Interop.UIWorkspace_Add(mInstance, aDocument.Instance);
        }
    }
}
