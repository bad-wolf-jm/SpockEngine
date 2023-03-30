using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDialog : UIComponent
    {
        public UIDialog() : this(UIDialog_Create()) { }

        public UIDialog(ulong aDerived) : base(aDerived) { }

        ~UIDialog() { UIDialog_Destroy(mInstance); }

        public void SetTitle(string aTitle) { UIDialog_SetTitle(mInstance, aTitle); }

        public void SetContent(UIComponent aContent) { UIDialog_SetContent(mInstance, aContent.Instance); }

        public void SetSize(Math.vec2 aSize) { UIDialog_SetSize(mInstance, aSize); }

        public void Update() { UIDialog_Update(mInstance); }
        
        public void Open() { UIDialog_Open(mInstance); }
        
        public void Close() { UIDialog_Close(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDialog_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIDialog_CreateWithTitleAndSize(ulong aInstance, string aTitle, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetTitle(ulong aInstance, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetSize(ulong aInstance, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Update(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Open(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIDialog_Close(ulong aInstance);
    }
}
