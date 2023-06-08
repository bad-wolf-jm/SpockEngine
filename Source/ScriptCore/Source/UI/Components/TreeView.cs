using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Math = SpockEngine.Math;

namespace SpockEngine
{
    public class UITreeViewNode : UIComponent
    {
        public UITreeViewNode() { mInstance = IntPtr.Zero; }

        public UITreeViewNode(IntPtr aInstance) : base(aInstance) { }

        ~UITreeViewNode()
        {
            Interop.UITreeViewNode_Destroy(mInstance);
        }

        private UIComponent mIcon;
        private void SetIcon(UIImage aIcon)
        {
            mIcon = aIcon;
            Interop.UITreeViewNode_SetIcon(mInstance, aIcon.Instance);
        }

        private UIComponent mIndicator;
        private void SetIndicator(UIComponent aIcon)
        {
            mIndicator = aIcon;
            Interop.UITreeViewNode_SetIcon(mInstance, aIcon.Instance);
        }


        public void SetText(string aText)
        {
            Interop.UITreeViewNode_SetText(mInstance, aText);
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            Interop.UITreeViewNode_SetTextColor(mInstance, aColor);
        }

        List<UITreeViewNode> mChildren = new List<UITreeViewNode>();
        public UITreeViewNode Add()
        {
            var lNewChild = new UITreeViewNode(Interop.UITreeViewNode_Add(mInstance));
            mChildren.Add(lNewChild);

            return lNewChild;
        }
    }

    public class UITreeView : UIComponent
    {
        bool mDerived = false;
        public UITreeView() : this(Interop.UITreeView_Create(), false) { }
        public UITreeView(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UITreeView() { if (!mDerived) Interop.UITreeView_Destroy(mInstance); }

        public void SetIndent(float aIndent) { Interop.UITreeView_SetIndent(mInstance, aIndent); }
        public void SetIconSpacing(float aSpacing) { Interop.UITreeView_SetIconSpacing(mInstance, aSpacing); }

        List<UITreeViewNode> mChildren = new List<UITreeViewNode>();
        public UITreeViewNode Add()
        {
            var lNewChild = new UITreeViewNode(Interop.UITreeView_Add(mInstance));
            mChildren.Add(lNewChild);

            return lNewChild;
        }
    }
}
