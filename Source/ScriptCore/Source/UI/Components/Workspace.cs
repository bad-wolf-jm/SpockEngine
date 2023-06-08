using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIWorkspaceDocument : UIComponent
    {
        public delegate bool DocumentSaveDelegate();

        public UIWorkspaceDocument() : base(UIWorkspaceDocument_Create())
        {
            UIWorkspaceDocument_RegisterSaveDelegate(mInstance, DoSave);
        }

        public bool IsDirty
        {
            get { return UIWorkspaceDocument_IsDirty(mInstance); }
            set { UIWorkspaceDocument_MarkAsDirty(mInstance, value); }
        }

        public void Open() { UIWorkspaceDocument_IsDirty(mInstance); }

        public void RequestClose() { UIWorkspaceDocument_RequestClose(mInstance); }

        public void ForceClose() { UIWorkspaceDocument_ForceClose(mInstance); }

        public virtual bool DoSave() { return true; }
    }

    public class UIWorkspace : UIComponent
    {
        private List<UIWorkspaceDocument> mDocuments;

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

        ~UIWorkspace() { UIWorkspace_Destroy(mInstance); }

        public void Add(UIWorkspaceDocument aDocument)
        {
            mDocuments.Add(aDocument);

            UIWorkspace_Add(mInstance, aDocument.Instance);
        }
    }
}
