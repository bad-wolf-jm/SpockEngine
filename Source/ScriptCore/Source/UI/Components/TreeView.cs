using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using Math = SpockEngine.Math;

namespace SpockEngine
{
    public class UITreeViewNode : UIComponent
    {
        public UITreeViewNode() { mInstance = 0; }

        public UITreeViewNode(ulong aInstance) : base(aInstance) { }

        ~UITreeViewNode()
        {
            UITreeViewNode_Destroy(mInstance);
        }

        private void SetIcon(UIImage aIcon)
        {
            UITreeViewNode_SetIcon(mInstance, aIcon.Instance);
        }

        public void SetText(string aText)
        {
            UITreeViewNode_SetText(mInstance, aText);
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            UITreeViewNode_SetTextColor(mInstance, aColor);
        }

        public UITreeViewNode Add()
        {
            return new UITreeViewNode(UITreeViewNode_Add(mInstance));
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetText(ulong aInstance, string aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeViewNode_SetIcon(ulong aInstance, ulong aIcon);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeViewNode_Add(ulong aInstance);
    }

    public class UITreeView : UIComponent
    {
        public UITreeView() : base(UITreeView_Create()) { }

        ~UITreeView() { UITreeView_Destroy(mInstance); }

        public void SetIndent(float aIndent) { UITreeView_SetIndent(mInstance, aIndent); }

        public UITreeViewNode Add()
        {
            return new UITreeViewNode(UITreeView_Add(mInstance));
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeView_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeView_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITreeView_SetIndent(ulong aInstance, float aIndent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITreeView_Add(ulong aInstance);
    }
}
